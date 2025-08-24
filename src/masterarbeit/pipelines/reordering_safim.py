import argparse
import json
import os
import subprocess
import tempfile
import logging
from typing import Tuple, Optional, Dict, List, Set

from tqdm import tqdm
import astunparse
import gast
import re
# Import SAFIM modules
from safim.data_utils import load_dataset
from safim.model_utils import build_model
from safim.prompt_utils import apply_prompt, apply_postprocessors
from safim.ast_utils import ErrorCheckVisitor, get_parser
from safim.evaluate import syntax_match

# Import adversarial attack modules
from masterarbeit.attacks.reordering.perturbation import perturbation, PERTURBATION_PLACEHOLDER_STR_RAW, NodeFinder

logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTION FOR FILENAME SANITIZATION
# ============================================================================
def sanitize_filename(name: str) -> str:
    """Replaces characters not suitable for filenames with underscores."""
    name = name.replace('/', '_') # Replace slashes often found in model names
    name = re.sub(r'[^\w\.-_]', '_', name) # Replace other non-alphanumeric (except ., -, _)
    return name

# ============================================================================
# STEP 1: Core evaluation functions (from original script)
# ============================================================================

def check_python_syntax(code_string: str) -> bool:
    """Check if Python code has valid syntax."""
    parser = get_parser("python")
    tree = parser.parse(code_string.encode("utf-8"))
    error_check = ErrorCheckVisitor()
    error_check(tree)
    return error_check.error_cnt == 0

def run_python_tests(code_string: str, unit_tests: list, task_id: str) -> Tuple[str, bool, int, int, int]:
    """
    Run unit tests for Python code.
    Returns: (result_str, overall_passed, tests_passed_count, tests_failed_count, total_tests)
    """
    if not unit_tests:
        return "NO_TESTS_APPLICABLE", True, 0, 0, 0

    if not check_python_syntax(code_string):
        logger.warning(f"Task {task_id}: Syntax error before test run.")
        return "COMPILATION_ERROR", False, 0, len(unit_tests), len(unit_tests)

    passed_count = 0
    failed_count = 0
    total_tests = len(unit_tests)

    for i, test_case in enumerate(unit_tests):
        test_input = test_case.get("input", "")
        expected_outputs = [str(o).strip() for o in test_case.get("output", [])]


        temp_file_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False, encoding='utf-8') as tmp_f:
                tmp_f.write(code_string)
                temp_file_path = tmp_f.name
            
            logger.debug(f"Task {task_id}, Test {i+1}/{total_tests}: Running with input: '{str(test_input)[:100]}'")
            process = subprocess.run(
                ["python", temp_file_path], 
                input=str(test_input), 
                text=True,
                capture_output=True, 
                timeout=10  
            )

            if process.returncode != 0:
                logger.warning(f"Task {task_id}, Test {i+1}/{total_tests}: RUNTIME_ERROR. Stderr: {process.stderr[:300]}")
                failed_count += 1
                continue
            
            actual_stdout_lines = [line.strip() for line in process.stdout.strip().splitlines()]
            if actual_stdout_lines == expected_outputs and process.returncode == 0: 
                passed_count += 1
            else: # Passed execution but wrong answer
                logger.warning(f"Task {task_id}, Test {i+1}/{total_tests}: WRONG_ANSWER.")
                logger.debug(f"Expected: {expected_outputs}, Got: {actual_stdout_lines}")
                failed_count += 1
                
        except subprocess.TimeoutExpired:
            logger.warning(f"Task {task_id}, Test {i+1}/{total_tests}: TIMEOUT.")
            failed_count += 1
        except Exception as e:
            logger.error(f"Task {task_id}, Test {i+1}/{total_tests}: Error during local exec: {e}", exc_info=True)
            failed_count += 1
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)


    overall_passed = failed_count == 0
    final_status = "PASSED" if overall_passed else "FAILED_SOME_TESTS" # More descriptive
    if not overall_passed and passed_count == 0 and failed_count == total_tests and total_tests > 0 :
         final_status = "FAILED_ALL_TESTS"
    elif not overall_passed and total_tests > 0 :
         final_status = "FAILED_SOME_TESTS"


    return final_status, overall_passed, passed_count, failed_count, total_tests

def get_completion_type_from_dataset(dataset_name: str) -> str:
    """Map dataset names to completion types."""
    if dataset_name in ['block_v2', 'block']:
        return 'block'
    elif dataset_name in ['control_fixed', 'control']:
        return 'control'
    elif dataset_name == 'api':
        return 'api'
    elif dataset_name == 'statement':
        return 'statement'
    else:
        return dataset_name

# ============================================================================
# STEP 2: Adversarial attack functions (NEW)
# ============================================================================

def generate_adversarial_reordering(eval_prompt: str, ground_truth: str, _precomputed_gt_strings: Optional[List[str]] = None,
                                    _precomputed_gt_types: Optional[Set[type]] = None) -> Optional[str]:
    """
    Generate a single adversarial reordering of the code.
    Takes the eval_prompt with {{completion}} placeholder and ground truth,
    creates complete code, reorders it, then replaces ground truth back with placeholder.
    
    Returns: Reordered incomplete code with {{completion}} placeholder, or None if fails
    """
    try:
        full_code = eval_prompt.replace("{{completion}}", ground_truth)
        
        # Pass logger to perturbation function
        reordered_full_code, _ = perturbation( 
            code=full_code,
            original_eval_prompt_template=eval_prompt, 
            ground_truth_code_str=ground_truth,        
            apply_perturbation=True,
            apply_renaming=False,
            logger_instance=logger,
            _precomputed_normalized_gt_strings=_precomputed_gt_strings,
            _precomputed_gt_node_types=_precomputed_gt_types
        )
        
        if reordered_full_code is None or reordered_full_code == full_code: # if perturbation failed or made no change
            logger.debug("Perturbation returned None or identical code.")
            return None

        try:
            compile(reordered_full_code, '<string>', 'exec')
        except SyntaxError:
            logger.warning(f"Perturbed full code has syntax error:\n{reordered_full_code[:500]}...")
            return None

        unparsed_placeholder_str = astunparse.unparse(gast.Expr(value=gast.Constant(value=PERTURBATION_PLACEHOLDER_STR_RAW, kind=None))).strip()

        # if ground_truth not in reordered_full_code:
        #     logger.warning("Ground truth not found in reordered full code. Cannot create adversarial prompt.")
        #     logger.debug(f"Ground Truth:\n{ground_truth[:500]}...")
        #     logger.debug(f"Reordered Code:\n{reordered_full_code[:500]}...")
        #     return None

        if unparsed_placeholder_str not in reordered_full_code:
            logger.warning(f"Ground truth placeholder '{unparsed_placeholder_str}' (from '{PERTURBATION_PLACEHOLDER_STR_RAW}') not found in reordered full code. Cannot create adversarial prompt.")
            logger.debug(f"Ground Truth (original for IDing nodes):\n{ground_truth[:300]}...")
            logger.debug(f"Reordered Code (should contain placeholder):\n{reordered_full_code[:500]}...")
            return None 
            
        reordered_incomplete = reordered_full_code.replace(unparsed_placeholder_str, "{{completion}}", 1)

        if reordered_incomplete == eval_prompt.replace("{{completion}}", "{{completion}}"): # Check if it's actually different
            logger.debug("Reordered incomplete prompt is identical to original incomplete prompt.")
            return None

        return reordered_incomplete.strip()
        
    except Exception as e:
        logger.error(f"Reordering failed unexpectedly: {e}", exc_info=True)
        return None

def find_adversarial_example(
    sample: dict,
    model,
    completion_type: str,
    mode: str,
    post_processors: list,
    cache: dict,
    max_attempts: int = 10
) -> Tuple[Optional[str], bool, int, Optional[Dict], Optional[str], str]: # Added str for attack_outcome_status
    task_id = sample.get("task_id", "unknown")
    eval_prompt = sample.get("eval_prompt", "")  
    ground_truth = sample.get("ground_truth", "")
    unit_tests = sample.get("unit_tests", [])

    logger.debug(f"Task {task_id}: Pre-normalizing ground truth for matching: '{ground_truth[:100]}...'")
    gt_normalizer = NodeFinder(ground_truth, logger_instance=logger) # from masterarbeit.attacks.reordering.perturbation import NodeFinder
    precomputed_gt_strings = gt_normalizer.target_strings_normalized
    precomputed_gt_types = gt_normalizer.target_node_types

    if not precomputed_gt_strings:
        logger.warning(f"Task {task_id}: Ground truth normalization failed (no target strings produced). "
                       "Cannot reliably identify GT nodes for perturbation. Skipping adversarial search.")
        return None, False, 0, None, None, "SETUP_FAILED_GT_NORMALIZATION"
    logger.debug(f"Task {task_id}: Pre-normalized GT into {len(precomputed_gt_strings)} parts with types {precomputed_gt_types}.")

    at_least_one_reordering_generated_and_placeholder_found = False # Keep this
    for attempt in range(max_attempts):
        logger.debug(f"Task {task_id}: Adversarial attempt {attempt + 1}/{max_attempts}")



    
    if not eval_prompt or not ground_truth or not unit_tests:
        logger.warning(f"Task {task_id}: Missing required fields for adv. search. Skipping.")
        return None, False, 0, None, None, "SKIPPED_PRE_CONDITIONS" # Indicate skip reason
    
    logger.info(f"Task {task_id}: Searching for adversarial example.")
    
    at_least_one_reordering_generated_and_placeholder_found = False
    for attempt in range(max_attempts):
        logger.debug(f"Task {task_id}: Adversarial attempt {attempt + 1}/{max_attempts}")
        reordered_prompt = generate_adversarial_reordering(
            eval_prompt, 
            ground_truth,
            _precomputed_gt_strings=precomputed_gt_strings, # NEW
            _precomputed_gt_types=precomputed_gt_types     # NEW
        )
        
        if reordered_prompt is None: # This means placeholder step likely failed
            logger.debug(f"Task {task_id}, Attempt {attempt+1}: generate_adversarial_reordering returned None (placeholder/setup issue).")
            # Don't set at_least_one_reordering_generated_and_placeholder_found to True here
            continue # Try next attempt if any
        
        at_least_one_reordering_generated_and_placeholder_found = True # A valid reordered prompt was made
        
        adv_sample = sample.copy()
        adv_sample["eval_prompt"] = reordered_prompt 
        
        prompt_for_model = apply_prompt(adv_sample, completion_type, mode, model)
        raw_completion = model.invoke_cached(prompt_for_model, cache)
        
        try:
            adv_completion = apply_postprocessors(
                raw_completion, adv_sample, completion_type, post_processors
            )
        except Exception as e: # Catch any postprocessing error
            logger.error(f"Task {task_id}, Adv Attempt {attempt+1}: Postprocessing failed. Raw: '{raw_completion[:100]}'. Error: {e}", exc_info=True)
            continue # Consider this attempt failed to produce a testable completion

        if adv_completion.strip() == ground_truth.strip():
            logger.debug(f"Task {task_id}, Adv Attempt {attempt+1}: Adversarial completion matches ground truth.")
            continue # Not an attack if it's GT
        
        adv_full_code_for_testing = eval_prompt.replace("{{completion}}", adv_completion)
        logger.debug(f"Task {task_id}, Adv Attempt {attempt+1}: Testing adversarial completion in ORIGINAL template.")
        result_str, passed_bool, p_tests, f_tests, t_tests = run_python_tests(
            adv_full_code_for_testing, unit_tests, f"{task_id}_adv_{attempt}"
        )
        adv_test_details = {
            "tests_passed": p_tests, "tests_failed": f_tests, "tests_total": t_tests,
            "status": result_str, "tested_in_original_template": True
        }

        if not passed_bool and t_tests > 0: 
            logger.warning(f"Task {task_id}: FOUND ADVERSARIAL EXAMPLE after {attempt + 1} attempts.")
            # ... (existing detailed logging for found example) ...
            return reordered_prompt, True, attempt + 1, adv_test_details, adv_completion, "FOUND_ADVERSARIAL"
    
    # Loop finished
    if not at_least_one_reordering_generated_and_placeholder_found:
        logger.info(f"Task {task_id}: No valid reordering prompt (placeholder setup) could be generated after {max_attempts} attempts.")
        return None, False, max_attempts, None, None, "SETUP_FAILED_REORDERING"
    else: # Reorderings were generated and tested, but none were effective adversarial examples
        logger.info(f"Task {task_id}: No effective adversarial example found after {max_attempts} attempts (all reorderings tested were non-adversarial or matched GT).")
        return None, False, max_attempts, None, None, "NO_EFFECTIVE_REORDERING"

# ============================================================================
# STEP 3: Main evaluation function with adversarial attack
# ============================================================================

def evaluate_with_adversarial_attack(
    sample: dict,
    model,
    completion_type: str,
    mode: str,
    post_processors: list,
    cache: dict,
    adversarial_attempts: int = 10,
    skip_adversarial: bool = False
) -> dict:
    """Main evaluation function with adversarial attack."""
    task_id = sample.get("task_id", "unknown")
    result: Dict[str, any] = {"task_id": task_id, "lang": sample.get("lang")}
    
    try:
        # First, get the model's completion on the original (non-reordered) code
        prompt = apply_prompt(sample, completion_type, mode, model)
        raw_completion = model.invoke_cached(prompt, cache)
        completion = apply_postprocessors(
            raw_completion, 
            sample, 
            completion_type, 
            post_processors
        )
        result["completion"] = completion
        
        if sample.get("lang") == "python" and sample.get("unit_tests"):
            full_code = sample["eval_prompt"].replace("{{completion}}", completion)
            
            # Test original completion
            if completion.strip() == sample.get("ground_truth", "").strip():
                result["original_result_status"] = "PASSED_GT_MATCH"
                result["original_passed"] = True
                result["original_tests_passed"] = len(sample["unit_tests"])
                result["original_tests_failed"] = 0
                result["original_tests_total"] = len(sample["unit_tests"])
            else:
                status_str, passed_bool, p_tests, f_tests, t_tests = run_python_tests(
                    full_code, sample["unit_tests"], task_id
                )
                result["original_result_status"] = status_str
                result["original_passed"] = passed_bool
                result["original_tests_passed"] = p_tests
                result["original_tests_failed"] = f_tests
                result["original_tests_total"] = t_tests

            # Adversarial attack - ONLY if original passed
            result["adversarial_skipped"] = skip_adversarial
            
            result["adversarial_skipped_reason"] = None # Initialize
            if result["original_passed"] and not skip_adversarial and sample.get("unit_tests") and sample.get("ground_truth"):
                logger.info(f"Attempting adversarial attack for {task_id} (original passed)")
                
                adv_prompt, found_adv, num_att, adv_test_details, adv_completion, adv_search_status = find_adversarial_example(
                    sample, model, completion_type, mode, post_processors, cache, adversarial_attempts
                )
                
                result["adversarial_found"] = found_adv
                result["adversarial_attempts_made"] = num_att

                if adv_search_status == "FOUND_ADVERSARIAL":
                    result["adversarial_prompt"] = adv_prompt
                    result["adversarial_completion"] = adv_completion
                    result["adversarial_status"] = "VULNERABLE"
                    result["adversarial_test_details"] = adv_test_details
                    logger.warning(f"Adversarial attack succeeded for {task_id}. Model failed {adv_test_details.get('tests_failed',0)} tests on reordered code.")
                elif adv_search_status == "SETUP_FAILED_REORDERING":
                    result["adversarial_status"] = "ATTACK_SETUP_FAILED"
                elif adv_search_status == "NO_EFFECTIVE_REORDERING":
                    result["adversarial_status"] = "ROBUST"
                
            elif not result["original_passed"]:
                # Original failed, so we skip adversarial attack
                result["adversarial_status"] = "SKIPPED_ORIGINAL_FAILED"
                result["adversarial_skipped_reason"] = "Original completion failed tests."
                result["adversarial_found"] = False
            elif skip_adversarial:
                result["adversarial_status"] = "SKIPPED_MANUALLY"
                result["adversarial_skipped_reason"] = "Adversarial attack manually skipped via CLI."
                result["adversarial_found"] = False
            elif not sample.get("unit_tests") or not sample.get("ground_truth"):
                result["adversarial_status"] = "SKIPPED_NO_TESTS_OR_GT"
                result["adversarial_skipped_reason"] = "Missing unit tests or ground truth for adversarial setup."
                result["adversarial_found"] = False
            else: # Should not be reached if logic is complete, but as a fallback
                result["adversarial_status"] = "SKIPPED_UNKNOWN_REASON"
                result["adversarial_found"] = False
                
        else:  # Non-Python or no unit tests
            if syntax_match(completion, sample.get("ground_truth", ""), sample.get("lang", "python")):
                result["original_result_status"] = "EXACT_MATCH"
                result["original_passed"] = True
            else:
                result["original_result_status"] = "WRONG_ANSWER_SYNTAX_MATCH"
                result["original_passed"] = False
            
            result["adversarial_status"] = "NOT_APPLICABLE_NON_PY_OR_NO_TESTS"
            result["adversarial_found"] = False
            
    except Exception as e:
        logger.error(f"Error during evaluation for {task_id}: {e}", exc_info=True)
        result["error_in_evaluation"] = str(e)
        result["original_result_status"] = "EVALUATION_ERROR"
        result["original_passed"] = False
        result["adversarial_status"] = "EVALUATION_ERROR_NO_ADV_ATTEMPT"
        result["adversarial_found"] = False
    
    return result

# ============================================================================
# STEP 4: Main function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SAFIM Evaluation with Adversarial Attack")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-coder-1.3b-instruct",
                        help="Model name (e.g., deepseek-ai/deepseek-coder-1.3b-instruct)")
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default='block',
                        choices=['api', 'block', 'block_v2', 'statement', 'control', 'control_fixed'],
                        help="Dataset to use")
    
    # Generation settings
    parser.add_argument("--mode", type=str, default="infilling",
                        choices=["infilling", "reverse_infilling", "left_to_right", 
                                "prefix_feeding", "instructed", "fewshot"],
                        help="Prompt mode")
    
    parser.add_argument("--post_processors", type=str, nargs="+",
                        help="Post-processors to apply")
    
    # Adversarial settings
    parser.add_argument("--adversarial_attempts", type=int, default=20,
                        help="Maximum adversarial reordering attempts per sample")
    
    parser.add_argument("--skip_adversarial", action="store_true",
                        help="Skip adversarial attack (for baseline comparison)")
    
    # Other arguments
    parser.add_argument("--base_output_dir", type=str, default="./safim_outputs",
                        help="Base directory to save logs and results.")
    parser.add_argument("--cache_path", type=str, default="./model_cache.json", 
                        help="Path to model cache")
    parser.add_argument("--sample_limit", type=int, default=0,
                        help="Limit number of samples (0 for no limit)")
    parser.add_argument("--python_only", default=True,
                        help="Only process Python samples")
    parser.add_argument("--block_comments", action="store_true",
                        help="Block comment generation") 
    
    args = parser.parse_args()

    # --- Path Setup ---
    sanitized_model_name = sanitize_filename(args.model_name)
    
    log_dir = os.path.join(args.base_output_dir, "Log")
    results_dir = os.path.join(args.base_output_dir, "Results")
    summary_dir = os.path.join(args.base_output_dir, "Summary")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    dynamic_log_file_path = os.path.join(log_dir, f"{sanitized_model_name}.log")
    dynamic_results_file_path = os.path.join(results_dir, f"{sanitized_model_name}.jsonl")
    dynamic_summary_file_path = os.path.join(summary_dir, f"{sanitized_model_name}_summary.txt")

    # --- Logger Reconfiguration ---
    # Remove existing handlers if any were added by default
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO, # Or INFO for less verbosity
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=[
            logging.FileHandler(dynamic_log_file_path, mode='w'), # Overwrite log file each run
            logging.StreamHandler()
        ]
    )
    logger.info(f"Log file will be saved to: {dynamic_log_file_path}")
    logger.info(f"Results file will be saved to: {dynamic_results_file_path}")
    logger.info(f"Summary file will be saved to: {dynamic_summary_file_path}")
    
    # Set default post-processors
    if args.post_processors is None:
        completion_type_for_pp = get_completion_type_from_dataset(args.dataset_name) # Renamed to avoid conflict
        if completion_type_for_pp == 'control':
            args.post_processors = ['truncate_control']
        elif completion_type_for_pp == 'api':
            args.post_processors = ['truncate_api_call']
        elif completion_type_for_pp in ['block', 'block_v2']:
            args.post_processors = ['truncate_line_until_block']
        elif completion_type_for_pp == 'statement':
            args.post_processors = ['truncate_line']
        else:
            args.post_processors = []
        
        logger.info(f"Using default post-processors for '{completion_type_for_pp}': {args.post_processors}")
    
    # Load cache
    cache = {}
    if os.path.exists(args.cache_path):
        try:
            with open(args.cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            logger.info(f"Loaded cache with {len(cache)} entries from {args.cache_path}.")
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")

    # Build model
    logger.info(f"Building model: {args.model_name}")
    model_wrapper = build_model(args)

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)
    
    # Get the correct completion type for prompting
    completion_type = get_completion_type_from_dataset(args.dataset_name)
    logger.info(f"Using completion type: {completion_type}")
    
    # Filter to Python samples if specified
    if args.python_only:
        dataset = [s for s in dataset if s.get("lang") == "python"]
        logger.info(f"Filtered to {len(dataset)} Python samples")
    
    # Apply sample limit
    if args.sample_limit > 0:
        dataset = dataset[:args.sample_limit]
        logger.info(f"Limited to {args.sample_limit} samples")

    # Process samples
    results = []
    total_samples_processed = 0
    original_pass_count = 0
    evaluation_error_count = 0 # Counts major errors in evaluate_with_adversarial_attack
    adversarial_vulnerable_count = 0 
    adversarial_robust_count = 0     
    adversarial_skipped_orig_failed_count = 0 
    adversarial_skipped_other_reasons_count = 0 
    adversarial_setup_failed_count = 0
    
    total_original_tests_attempted = 0
    total_original_tests_passed = 0
    total_adv_tests_attempted_on_vulnerable = 0 
    total_adv_tests_failed_on_vulnerable = 0    
    
    for sample_idx, sample in enumerate(tqdm(dataset, desc="Processing samples")): # Use enumerate for periodic save
        total_samples_processed +=1
        eval_result = evaluate_with_adversarial_attack(
            sample, 
            model_wrapper, 
            completion_type,
            args.mode,
            args.post_processors,
            cache,
            adversarial_attempts=args.adversarial_attempts,
            skip_adversarial=args.skip_adversarial
        )
        results.append(eval_result)

        if eval_result.get("error_in_evaluation"):
            evaluation_error_count += 1
        elif eval_result.get("original_passed"):
            original_pass_count += 1 # This is a sample eligible for adv attack
            
            adv_status = eval_result.get("adversarial_status")
            if adv_status == "VULNERABLE": 
                adversarial_vulnerable_count += 1
                adv_details = eval_result.get("adversarial_test_details")
                if adv_details:
                    total_adv_tests_attempted_on_vulnerable += adv_details.get("tests_total",0)
                    total_adv_tests_failed_on_vulnerable += adv_details.get("tests_failed",0)
            elif adv_status == "ROBUST":
                adversarial_robust_count += 1
            elif adv_status == "ATTACK_SETUP_FAILED":
                adversarial_setup_failed_count += 1
            # SKIPPED_ORIGINAL_FAILED is handled in the 'else' for original_passed
            # SKIPPED_MANUALLY / SKIPPED_NO_TESTS_OR_GT etc. are also distinct now
            elif adv_status not in ["SKIPPED_ORIGINAL_FAILED", "SKIPPED_MANUALLY", "SKIPPED_NO_TESTS_OR_GT", "NOT_APPLICABLE_NON_PY_OR_NO_TESTS"]:
                adversarial_skipped_other_reasons_count +=1 # Catch-all for other skips on original passes

        else: # Original failed
            if eval_result.get("adversarial_status") == "SKIPPED_ORIGINAL_FAILED":
                adversarial_skipped_orig_failed_count += 1
            elif eval_result.get("adversarial_status") not in ["VULNERABLE", "ROBUST"]: # Catch other skips when original failed
                adversarial_skipped_other_reasons_count +=1


        if sample.get("lang") == "python" and sample.get("unit_tests") and not eval_result.get("error_in_evaluation"):
            total_original_tests_attempted += eval_result.get("original_tests_total", 0)
            total_original_tests_passed += eval_result.get("original_tests_passed", 0)
        
        if (sample_idx + 1) % 50 == 0: 
            with open(args.cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
            logger.info(f"Saved cache at sample {sample_idx + 1}")
    
    with open(args.cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    logger.info(f"Final cache saved with {len(cache)} entries.")
    
    with open(dynamic_results_file_path, "w", encoding="utf-8") as f: 
        for r_item in results:
            f.write(json.dumps(r_item) + "\n")
    logger.info(f"Results saved to: {dynamic_results_file_path}") 
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Results Summary:")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples processed: {total_samples_processed}")
    logger.info(f"Samples with major evaluation errors (aborted eval for sample): {evaluation_error_count}")

    num_evaluable = total_samples_processed - evaluation_error_count
    logger.info(f"\n--- Original Completion Performance (on {num_evaluable} evaluable samples) ---")
    if num_evaluable > 0:
        logger.info(f"Passed (original prompt): {original_pass_count} ({original_pass_count / num_evaluable * 100:.2f}%)")
    else:
        logger.info("Passed (original prompt): 0 (N/A)")
        
    if total_original_tests_attempted > 0:
        logger.info(f"  Total unit tests (original): {total_original_tests_attempted}")
        logger.info(f"  Tests passed (original): {total_original_tests_passed} ({total_original_tests_passed / total_original_tests_attempted * 100:.2f}%)")

    logger.info(f"\n--- Adversarial Attack Results ---")
    num_eligible_for_adv = original_pass_count # Samples where original completion passed
    logger.info(f"Samples eligible for adversarial attack (original passed tests): {num_eligible_for_adv}")

    logger.info(f"  Vulnerable (attack succeeded): {adversarial_vulnerable_count}")
    logger.info(f"  Robust (attack attempted, no adversarial effect found): {adversarial_robust_count}")
    logger.info(f"  Attack Setup Failed (placeholder/reordering issues prevented test): {adversarial_setup_failed_count}")

    num_adv_attacks_actually_run = adversarial_vulnerable_count + adversarial_robust_count
    if num_adv_attacks_actually_run > 0:
        logger.info(f"  Vulnerability Rate (among successfully setup and tested attacks): {adversarial_vulnerable_count / num_adv_attacks_actually_run * 100:.2f}%")
        logger.info(f"  Robustness Rate (among successfully setup and tested attacks): {adversarial_robust_count / num_adv_attacks_actually_run * 100:.2f}%")
    else:
        logger.info("  Vulnerability Rate (among successfully setup and tested attacks): N/A (no attacks successfully set up and tested)")
        logger.info("  Robustness Rate (among successfully setup and tested attacks): N/A (no attacks successfully set up and tested)")

    logger.info(f"Adversarial attack skipped because original completion failed: {adversarial_skipped_orig_failed_count}")

    # More detailed skip reasons:
    adv_skipped_manually = sum(1 for r in results if r.get("adversarial_status") == "SKIPPED_MANUALLY")
    adv_skipped_no_gt_tests = sum(1 for r in results if r.get("adversarial_status") == "SKIPPED_NO_TESTS_OR_GT")
    adv_not_applicable = sum(1 for r in results if r.get("adversarial_status") == "NOT_APPLICABLE_NON_PY_OR_NO_TESTS")

    logger.info(f"Adversarial attack skipped for other reasons on original passes (e.g., manual, no GT/tests): {adv_skipped_manually + adv_skipped_no_gt_tests + adversarial_skipped_other_reasons_count}")
    logger.info(f"    Specifically: Manually skipped: {adv_skipped_manually}")
    logger.info(f"    Specifically: Skipped due to no GT/tests: {adv_skipped_no_gt_tests}")
    logger.info(f"    Specifically: Not applicable (non-Python/no tests): {adv_not_applicable}")

    if total_adv_tests_attempted_on_vulnerable > 0:
        logger.info(f"\nFor VULNERABLE samples:")
        logger.info(f"  Total adversarial tests attempted: {total_adv_tests_attempted_on_vulnerable}")
        logger.info(f"  Total adversarial tests failed: {total_adv_tests_failed_on_vulnerable} ({total_adv_tests_failed_on_vulnerable / total_adv_tests_attempted_on_vulnerable * 100:.2f}%)")
        if adversarial_vulnerable_count > 0: # Avoid division by zero
             logger.info(f"  Average tests failed per vulnerable sample: {total_adv_tests_failed_on_vulnerable / adversarial_vulnerable_count:.2f} / {total_adv_tests_attempted_on_vulnerable / adversarial_vulnerable_count:.2f}")


    with open(dynamic_summary_file_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(f"Model: {args.model_name}\n")
        summary_file.write(f"Dataset: {args.dataset_name}\n")
        summary_file.write(f"Mode: {args.mode}\n")
        summary_file.write(f"Post-processors: {', '.join(args.post_processors)}\n")
        summary_file.write(f"Adversarial Attempts: {args.adversarial_attempts}\n")
        summary_file.write(f"Skip Adversarial: {args.skip_adversarial}\n\n")

        summary_file.write(f"Total samples processed: {total_samples_processed}\n")
        summary_file.write(f"Samples with major evaluation errors (aborted eval for sample): {evaluation_error_count}\n\n")

        summary_file.write(f"Original Completion Performance (on {num_evaluable} evaluable samples):\n")
        if num_evaluable > 0:
            summary_file.write(f"  Passed (original prompt): {original_pass_count} ({original_pass_count / num_evaluable * 100:.2f}%)\n")
        else:
            summary_file.write("  Passed (original prompt): 0 (N/A)\n")

        if total_original_tests_attempted > 0:
            summary_file.write(f"    Total unit tests (original): {total_original_tests_attempted}\n")
            summary_file.write(f"    Tests passed (original): {total_original_tests_passed} ({total_original_tests_passed / total_original_tests_attempted * 100:.2f}%)\n")

        summary_file.write("\nAdversarial Attack Results:\n")
        summary_file.write(f"  Samples eligible for adversarial attack (original passed tests): {num_eligible_for_adv}\n")
        summary_file.write(f"  Vulnerable (attack succeeded): {adversarial_vulnerable_count}\n")
        summary_file.write(f"  Robust (attack attempted, no adversarial effect found): {adversarial_robust_count}\n")
        summary_file.write(f"  Attack Setup Failed (placeholder/reordering issues prevented test): {adversarial_setup_failed_count}\n")

        if num_adv_attacks_actually_run > 0:
            summary_file.write(f"  Vulnerability Rate (among successfully setup and tested attacks): {adversarial_vulnerable_count / num_adv_attacks_actually_run * 100:.2f}%\n")
            summary_file.write(f"  Robustness Rate (among successfully setup and tested attacks): {adversarial_robust_count / num_adv_attacks_actually_run * 100:.2f}%\n")
        else:
            summary_file.write("  Vulnerability Rate (among successfully setup and tested attacks): N/A (no attacks successfully set up and tested)\n")
            summary_file.write("  Robustness Rate (among successfully setup and tested attacks): N/A (no attacks successfully set up and tested)\n")
        
        summary_file.write(f"Adversarial attack skipped because original completion failed: {adversarial_skipped_orig_failed_count}\n")
        summary_file.write(f"Adversarial attack skipped for other reasons on original passes (e.g., manual, no GT/tests): {adv_skipped_manually + adv_skipped_no_gt_tests + adversarial_skipped_other_reasons_count}\n")
        summary_file.write(f"    Specifically: Manually skipped: {adv_skipped_manually}\n")
        summary_file.write(f"    Specifically: Skipped due to no GT/tests: {adv_skipped_no_gt_tests}\n")
        summary_file.write(f"    Specifically: Not applicable (non-Python/no tests): {adv_not_applicable}\n")

        if total_adv_tests_attempted_on_vulnerable > 0:
            summary_file.write(f"\nFor VULNERABLE samples:\n")
            summary_file.write(f"  Total adversarial tests attempted: {total_adv_tests_attempted_on_vulnerable}\n")
            summary_file.write(f"  Total adversarial tests failed: {total_adv_tests_failed_on_vulnerable} ({total_adv_tests_failed_on_vulnerable / total_adv_tests_attempted_on_vulnerable * 100:.2f}%)\n")
            if adversarial_vulnerable_count > 0: # Avoid division by zero
                summary_file.write(f"  Average tests failed per vulnerable sample: {total_adv_tests_failed_on_vulnerable / adversarial_vulnerable_count:.2f} / {total_adv_tests_attempted_on_vulnerable / adversarial_vulnerable_count:.2f}\n")

    logger.info(f"Summary saved to: {dynamic_summary_file_path}")

    logger.info(f"\nLog saved to: {dynamic_log_file_path}")
    
if __name__ == "__main__":
    main()