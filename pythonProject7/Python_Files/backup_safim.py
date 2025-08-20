import argparse
import gast
import ast
import astunparse
import json
import os
import subprocess
import tempfile
import logging
import re # Added import
from typing import Tuple, Optional, Dict, List

from tqdm import tqdm

# Import SAFIM modules
from safim.data_utils import load_dataset
from safim.model_utils import build_model
from safim.prompt_utils import apply_prompt, apply_postprocessors
from safim.ast_utils import ErrorCheckVisitor, get_parser
from safim.evaluate import syntax_match

# Import adversarial attack modules
from perturbation import perturbation # Assuming perturbation.py is in the same directory or accessible

# Define the 11 problematic samples where completion == adversarial_completion but marked VULNERABLE
PROBLEMATIC_TASK_IDS = [
    "block_completion_000075",
    "block_completion_000078",
    "block_completion_000079",
    "block_completion_000109",
    "block_completion_000432",
    "block_completion_000487",
    "block_completion_000488",
    "block_completion_000546",
    "block_completion_000732",
    "block_completion_000796",
    "block_completion_000799"
]

# Setup logging
LOG_FILE_PATH_LESS = "safim_adversarial_debug_11_samples_less.txt"
LOG_FILE_PATH_MORE = "safim_adversarial_debug_11_samples_more.txt"

if os.path.exists(LOG_FILE_PATH_LESS):
    os.remove(LOG_FILE_PATH_LESS)

if os.path.exists(LOG_FILE_PATH_MORE):
    os.remove(LOG_FILE_PATH_MORE)

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Set logger to the lowest level for capturing everything

# Create file handler which logs even debug messages
fh_less = logging.FileHandler(LOG_FILE_PATH_LESS)
fh_less.setLevel(logging.INFO)

fh_more = logging.FileHandler(LOG_FILE_PATH_MORE)
fh_more.setLevel(logging.DEBUG)

# Create console handler with a higher log level (INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO) # Console will only show INFO, WARNING, ERROR, CRITICAL

# Create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")
fh_less.setFormatter(formatter)
fh_more.setFormatter(formatter)
ch.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(fh_less)
logger.addHandler(fh_more)
logger.addHandler(ch)


logger.info("=" * 80)
logger.info("DEBUGGING 11 PROBLEMATIC SAMPLES")
logger.info("These samples have identical completions but different test results")
logger.info("Log file with DEBUG details: %s", LOG_FILE_PATH_LESS)
logger.info("=" * 80)

# ============================================================================
# STEP 1: Core evaluation functions
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
        logger.warning(f"Task {task_id}: Syntax error in generated code before test run.")
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
            else: 
                logger.warning(f"Task {task_id}, Test {i+1}/{total_tests}: WRONG_ANSWER.")
                logger.debug(f"Expected: {expected_outputs}, Got: {actual_stdout_lines}") # DEBUG for file
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
    final_status = "PASSED" if overall_passed else "FAILED_SOME_TESTS" 
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
# STEP 2: Adversarial attack helper functions
# ============================================================================

def find_ground_truth_flexible(full_code: str, ground_truth: str) -> Tuple[int, int, str]:
    """
    Enhanced ground truth detection with multiple strategies.
    Returns: (start_pos, end_pos, actual_text_found)
    """
    
    # Strategy 1: Direct string match
    pos = full_code.find(ground_truth)
    if pos != -1:
        return pos, pos + len(ground_truth), ground_truth
    
    # Strategy 2: Whitespace-flexible regex
    pattern = re.escape(ground_truth)
    pattern = pattern.replace(r'\ ', r'\s*')
    pattern = pattern.replace(r'\n', r'\s*\n\s*')
    
    match = re.search(pattern, full_code)
    if match:
        return match.start(), match.end(), match.group(0)
    
    # Strategy 3: For simple statements, try AST-based reconstruction
    if '\n' not in ground_truth and len(ground_truth) < 100:
        try:
            gt_tree = ast.parse(ground_truth.strip())
            if len(gt_tree.body) == 1:
                gt_stmt = gt_tree.body[0]
                full_tree = ast.parse(full_code)
                for node in ast.walk(full_tree):
                    if type(node) == type(gt_stmt):
                        if ast.dump(node) == ast.dump(gt_stmt):
                            if hasattr(node, 'lineno'):
                                lines = full_code.split('\n')
                                if 0 <= node.lineno - 1 < len(lines):
                                    line_start = sum(len(line) + 1 for line in lines[:node.lineno-1])
                                    reconstructed = astunparse.unparse(node).strip()
                                    search_start = max(0, line_start - 50)
                                    search_end = min(len(full_code), line_start + len(reconstructed) + 50)
                                    search_area = full_code[search_start:search_end]
                                    local_pos = search_area.find(reconstructed)
                                    if local_pos != -1:
                                        actual_start = search_start + local_pos
                                        return actual_start, actual_start + len(reconstructed), reconstructed
        except:
            pass 
    
    # Strategy 4: For multi-line ground truth, try line-based matching with flexibility
    if '\n' in ground_truth:
        gt_lines_stripped = [l.strip() for l in ground_truth.strip().split('\n') if l.strip()]
        if not gt_lines_stripped: return -1, -1, "" 

        full_lines = full_code.split('\n')
        full_lines_stripped = [l.strip() for l in full_lines]
        
        for i in range(len(full_lines_stripped) - len(gt_lines_stripped) + 1):
            match = True
            for j, gt_line_s in enumerate(gt_lines_stripped):
                if gt_line_s != full_lines_stripped[i + j]:
                    match = False
                    break
            
            if match:
                start_pos = sum(len(line) + 1 for line in full_lines[:i])
                matched_text_block = '\n'.join(full_lines[i : i + len(gt_lines_stripped)])
                end_pos = start_pos + len(matched_text_block)
                return start_pos, end_pos, matched_text_block
    
    return -1, -1, ""

def generate_adversarial_reordering(eval_prompt: str, ground_truth: str, sample: dict) -> Optional[str]:
    """
    Enhanced version with better ground truth handling.
    Uses find_ground_truth_flexible.
    """
    task_id = sample.get("task_id", "unknown_task")
    try:
        full_code = eval_prompt.replace("{{completion}}", ground_truth)

        logger.debug(f"Task {task_id}: Original eval_prompt for perturbation input:\n{eval_prompt}")
        logger.debug(f"Task {task_id}: Original ground_truth for perturbation input:\n{ground_truth}")
        
        reordered_full_code, _ = perturbation(
            full_code, 
            apply_perturbation=True, 
            apply_renaming=False,
            ground_truth_string=ground_truth,
            logger_instance=logger # Pass the logger instance
        )
        
        if reordered_full_code is None: # Perturbation itself might fail (e.g. cycle)
            logger.warning(f"Task {task_id}: Perturbation function returned None. Could not generate reordered code.")
            return None
                
        try:
            compile(reordered_full_code, '<string>', 'exec')
        except SyntaxError:
            logger.warning(f"Task {task_id}: Reordered code (full) has syntax error after perturbation. Code:\n{reordered_full_code}")
            return None
        
        gt_start, gt_end, actual_gt_found_in_reordered = find_ground_truth_flexible(reordered_full_code, ground_truth)
        
        if gt_start == -1:
            logger.warning(f"Task {task_id}: Ground truth NOT FOUND in reordered code using any strategy.")
            logger.debug(f"Original Ground truth for {task_id}: {repr(ground_truth[:200])}")
            logger.debug(f"Reordered full code (where GT was searched) for {task_id}:\n{reordered_full_code}")
            return None
        
        reordered_incomplete = (
            reordered_full_code[:gt_start] + 
            "{{completion}}" + 
            reordered_full_code[gt_end:]
        )

        logger.debug(f"Task {task_id}: Full code after perturbation by perturbation.py:\n{reordered_full_code}")
        
        if "{{completion}}" not in reordered_incomplete:
            logger.error(f"Task {task_id}: Failed to insert {{completion}} placeholder correctly in reordered code.")
            return None
        
        if re.sub(r'\s+', ' ', reordered_incomplete.strip()) == re.sub(r'\s+', ' ', eval_prompt.strip()):
            logger.info(f"Task {task_id}: No effective reordering occurred (reordered prompt is same as original).")
            return None 
        
        return reordered_incomplete.strip()
        
    except Exception as e:
        logger.error(f"Task {task_id}: Reordering process failed: {e}", exc_info=True)
        return None

def find_adversarial_example(
    sample: dict,
    model,
    completion_type: str,
    mode: str,
    post_processors: list,
    cache: dict,
    max_attempts: int = 10
) -> Tuple[Optional[str], bool, int, Optional[Dict], Optional[str], str]:
    task_id = sample.get("task_id", "unknown_task")
    eval_prompt = sample.get("eval_prompt", "")
    ground_truth = sample.get("ground_truth", "")
    unit_tests = sample.get("unit_tests", [])
    
    if not eval_prompt or not ground_truth or not unit_tests:
        logger.warning(f"Task {task_id}: Missing required fields (eval_prompt, ground_truth, or unit_tests) for adversarial attack.")
        return None, False, 0, None, None, "missing_fields"
    
    logger.info(f"Task {task_id}: Searching for adversarial example...")
    
    reordering_attempts_made = 0
    valid_reorderings_generated = 0
    model_passed_on_valid_reordering_count = 0
    
    for attempt_idx in range(max_attempts):
        reordering_attempts_made += 1
        logger.debug(f"Task {task_id}: Adversarial reordering attempt {attempt_idx + 1}/{max_attempts}")
        
        reordered_prompt = generate_adversarial_reordering(eval_prompt, ground_truth, sample)
        
        if reordered_prompt is None:
            logger.debug(f"Task {task_id}: Attempt {attempt_idx + 1} - reordered_prompt was None (GT not found, syntax error, or perturbation failed).")
            continue 
            
        valid_reorderings_generated += 1
        adv_sample = sample.copy()
        adv_sample["eval_prompt"] = reordered_prompt
        
        logger.debug(f"Task {task_id}: Attempt {attempt_idx + 1} - Getting model completion for valid reordered prompt.")
        
        prompt_for_model = apply_prompt(adv_sample, completion_type, mode, model)
        raw_completion = model.invoke_cached(prompt_for_model, cache)
        adv_completion = apply_postprocessors(
            raw_completion, 
            adv_sample, 
            completion_type, 
            post_processors
        )
        
        logger.info(f"Task {task_id}: Attempt {attempt_idx + 1} - Model's adversarial completion (first 50 chars): '{adv_completion[:50]}...'")
        
        adv_full_code = reordered_prompt.replace("{{completion}}", adv_completion)
        
        result_str, passed_bool, p_tests, f_tests, t_tests = run_python_tests(
            adv_full_code, unit_tests, f"{task_id}_adv_{attempt_idx}"
        )
        
        adv_test_details = {
            "tests_passed": p_tests,
            "tests_failed": f_tests,
            "tests_total": t_tests,
            "status": result_str
        }

        if not passed_bool and t_tests > 0: 
            logger.info(f"Task {task_id}: Found adversarial example after {reordering_attempts_made} reordering attempts ({valid_reorderings_generated} valid).")
            logger.info(f"  Model's completion (adversarial) failed {f_tests}/{t_tests} tests on reordered code.")
            return reordered_prompt, True, reordering_attempts_made, adv_test_details, adv_completion, "none"
        elif t_tests > 0: 
            model_passed_on_valid_reordering_count +=1
            logger.info(f"Task {task_id}: Attempt {attempt_idx + 1} - Model passed tests on this reordered prompt.")
        else:
             logger.info(f"Task {task_id}: Attempt {attempt_idx + 1} - No tests run or all tests were skipped for this reordered prompt's completion ({result_str}).")

    failure_reason = "unknown_after_max_attempts"
    if valid_reorderings_generated == 0:
        failure_reason = "no_valid_reorderings_generated_or_gt_not_found_all_attempts"
    elif model_passed_on_valid_reordering_count == valid_reorderings_generated:
        failure_reason = "model_robust_all_valid_reorderings_passed"
    elif model_passed_on_valid_reordering_count > 0 :
        failure_reason = "mixed_outcomes_no_adversary"
    else:
        failure_reason = "valid_reorderings_all_led_to_non_passing_non_failing_tests"

    logger.info(f"Task {task_id}: No adversarial example found after {max_attempts} reordering attempts.")
    logger.info(f"  Overall attempts: {reordering_attempts_made}, Valid reorderings generated: {valid_reorderings_generated}, Model passed on valid reorderings: {model_passed_on_valid_reordering_count}")
    logger.info(f"  Final failure reason: {failure_reason}")
    
    return None, False, reordering_attempts_made, None, None, failure_reason

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
    task_id = sample.get("task_id", "unknown_task")
    result: Dict[str, any] = {"task_id": task_id, "lang": sample.get("lang")}
    
    logger.info("\n" + "="*80 + f"\nDEBUGGING PROBLEMATIC SAMPLE: {task_id}\n" + "="*80)
    
    try:
        logger.info(f"Task {task_id}: Original eval_prompt:\n{sample['eval_prompt']}")
        prompt = apply_prompt(sample, completion_type, mode, model)
        raw_completion = model.invoke_cached(prompt, cache)
        completion = apply_postprocessors(
            raw_completion, 
            sample, 
            completion_type, 
            post_processors
        )
        result["completion"] = completion
        
        logger.info(f"Task {task_id}: Original model completion (full):")
        logger.info(f"------\n{completion}\n------")
        
        if sample.get("lang") == "python" and sample.get("unit_tests"):
            original_full_code = sample["eval_prompt"].replace("{{completion}}", completion)
            
            logger.info(f"\nORIGINAL FULL CODE (Prompt + Completion) FOR {task_id}:")
            logger.info("------------------------------------------------------------")
            logger.info(original_full_code)
            logger.info("------------------------------------------------------------")
            
            if sample.get("ground_truth") and completion.strip() == sample["ground_truth"].strip():
                result["original_result_status"] = "PASSED_GT_MATCH"
                result["original_passed"] = True
                num_tests = len(sample["unit_tests"]) if sample.get("unit_tests") else 0
                result["original_tests_passed"] = num_tests
                result["original_tests_failed"] = 0
                result["original_tests_total"] = num_tests
                logger.info(f"Task {task_id}: Original matches ground truth (implies all {num_tests} tests passed).")
            else:
                status_str, passed_bool, p_tests, f_tests, t_tests = run_python_tests(
                    original_full_code, sample["unit_tests"], task_id + "_original"
                )
                result["original_result_status"] = status_str
                result["original_passed"] = passed_bool
                result["original_tests_passed"] = p_tests
                result["original_tests_failed"] = f_tests
                result["original_tests_total"] = t_tests
                logger.info(f"Task {task_id}: Original code {'PASSED' if passed_bool else 'FAILED'} tests ({p_tests} passed / {t_tests} total). Status: {status_str}")

            should_skip_this_sample_adv = skip_adversarial or not result["original_passed"]
            result["adversarial_skipped_reason"] = "global_skip_true" if skip_adversarial else \
                                                  ("original_did_not_pass" if not result["original_passed"] else "not_skipped")
            
            if not should_skip_this_sample_adv and sample.get("ground_truth") and sample.get("unit_tests"):
                logger.info(f"Task {task_id}: Attempting adversarial attack as original passed and not globally skipped...")
                
                adv_prompt, found_adv, num_att, adv_test_details, adv_completion, failure_reason = find_adversarial_example(
                    sample, model, completion_type, mode, post_processors, cache, max_attempts=adversarial_attempts
                )
                
                result["adversarial_found"] = found_adv
                result["adversarial_attempts_made"] = num_att
                result["adversarial_failure_reason"] = failure_reason
                
                if found_adv:
                    result["adversarial_prompt"] = adv_prompt
                    result["adversarial_completion"] = adv_completion
                    result["adversarial_status"] = "VULNERABLE"
                    result["adversarial_test_details"] = adv_test_details
                    
                    logger.info(f"\nADVERSARIAL ATTACK SUCCEEDED for {task_id}:")
                    logger.info(f"  Original passed: {result['original_passed']}")
                    logger.info(f"  Adversarial completion (full):")
                    logger.info(f"------\n{adv_completion}\n------")
                    logger.info(f"  COMPLETIONS ARE {'IDENTICAL' if completion == adv_completion else 'DIFFERENT'}")
                    adv_failed_count = adv_test_details.get('tests_failed',0)
                    adv_total_count = adv_test_details.get('tests_total',0)
                    logger.info(f"  Adversarial code FAILED tests ({adv_failed_count} failed / {adv_total_count} total). Status: {adv_test_details.get('status')}")

                    adversarial_full_code = adv_prompt.replace("{{completion}}", adv_completion) # adv_prompt should be defined
                    logger.info(f"\nADVERSARIAL FULL CODE (Perturbed Prompt + Completion) FOR {task_id}:")
                    logger.info("------------------------------------------------------------")
                    logger.info(adversarial_full_code)
                    logger.info("------------------------------------------------------------")
                    
                    if completion == adv_completion:
                        logger.error(f"\n\n!!! CRITICAL ISSUE FOR {task_id} !!!")
                        logger.error("!!! IDENTICAL COMPLETION BUT DIFFERENT TEST RESULTS !!!")
                        logger.error("------------------------------------------------------------")
                        logger.error("ORIGINAL PROMPT (led to PASSING code):")
                        logger.error("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
                        logger.error(sample['eval_prompt'])
                        logger.error("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                        logger.error("\nADVERSARIAL PROMPT (led to FAILING code with SAME completion):")
                        logger.error("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
                        logger.error(adv_prompt) # adv_prompt should be defined
                        logger.error("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                        
                        logger.error("\nLINE-BY-LINE PROMPT DIFFERENCES (Original vs Adversarial):")
                        orig_lines = sample['eval_prompt'].split('\n')
                        adv_lines = adv_prompt.split('\n')
                        max_lines = max(len(orig_lines), len(adv_lines))
                        diff_found = False
                        for i in range(max_lines):
                            orig_line = orig_lines[i] if i < len(orig_lines) else "[LINE MISSING IN ORIGINAL]"
                            adv_line = adv_lines[i] if i < len(adv_lines) else "[LINE MISSING IN ADVERSARIAL]"
                            if orig_line != adv_line:
                                diff_found = True
                                logger.error(f"Line {i+1} DIFFERS:")
                                logger.error(f"  ORIGINAL : {orig_line}")
                                logger.error(f"  ADVERSARY: {adv_line}")
                        if not diff_found:
                            logger.error("No line-by-line differences found in prompts (whitespace or other subtle changes might exist).")
                        logger.error("------------------------------------------------------------\n\n")
                else:
                    result["adversarial_status"] = f"ATTACK_FAILED ({failure_reason})"
                    logger.info(f"Task {task_id}: Adversarial attack failed or conditions not met. Reason: {failure_reason}")

            elif should_skip_this_sample_adv:
                result["adversarial_status"] = "ADV_SKIPPED"
                result["adversarial_found"] = False
                logger.info(f"Task {task_id}: Adversarial attack skipped ({result['adversarial_skipped_reason']}).")
            else: # Missing ground_truth or unit_tests for attack
                result["adversarial_status"] = "ADV_PRECONDITIONS_NOT_MET_NO_GT_OR_TESTS"
                result["adversarial_found"] = False
                logger.info(f"Task {task_id}: Adversarial attack not applicable (missing ground_truth or unit_tests for attack).")
                
        else: 
            if sample.get("ground_truth") and syntax_match(completion, sample["ground_truth"], sample.get("lang", "python")):
                result["original_result_status"] = "EXACT_MATCH_SYNTAX"
                result["original_passed"] = True
            else:
                result["original_result_status"] = "NO_MATCH_SYNTAX_OR_NON_PY"
                result["original_passed"] = False
            logger.info(f"Task {task_id}: Non-Python or no unit tests. Original status: {result['original_result_status']}")
            result["adversarial_status"] = "NOT_APPLICABLE_NON_PY_OR_NO_TESTS"
            result["adversarial_found"] = False
            
    except Exception as e:
        logger.error(f"Task {task_id}: Error during evaluation: {e}", exc_info=True)
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
    parser = argparse.ArgumentParser(description="SAFIM Evaluation with Adversarial Attack - Debug Mode for 11 Samples")
    
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-coder-1.3b-instruct",
                        help="Model name (e.g., deepseek-ai/deepseek-coder-1.3b-instruct)")
    parser.add_argument("--dataset_name", type=str, default='block',
                        choices=['api', 'block', 'block_v2', 'statement', 'control', 'control_fixed'],
                        help="Dataset to use")
    parser.add_argument("--mode", type=str, default="infilling",
                        choices=["infilling", "reverse_infilling", "left_to_right", 
                                "prefix_feeding", "instructed", "fewshot"],
                        help="Prompt mode")
    parser.add_argument("--post_processors", type=str, nargs="+",
                        help="Post-processors to apply")
    parser.add_argument("--adversarial_attempts", type=int, default=1, # Reduced for quicker debugging focus
                        help="Maximum adversarial reordering attempts per sample")
    parser.add_argument("--skip_adversarial", action="store_true",
                        help="Skip adversarial attack (e.g., for baseline or if original fails)")
    parser.add_argument("--cache_path", type=str, default="./model_cache.json",
                        help="Path to model cache")
    parser.add_argument("--output_path", type=str, default="./results_adversarial_debug_11.jsonl",
                        help="Path to save results")
    parser.add_argument("--block_comments", action="store_true",
                        help="Block comment generation (parameter for build_model)")
    
    args = parser.parse_args()
    
    if args.post_processors is None:
        completion_type_for_pp = get_completion_type_from_dataset(args.dataset_name)
        if completion_type_for_pp == 'control': args.post_processors = ['truncate_control']
        elif completion_type_for_pp == 'api': args.post_processors = ['truncate_api_call']
        elif completion_type_for_pp in ['block', 'block_v2']: args.post_processors = ['truncate_line_until_block']
        elif completion_type_for_pp == 'statement': args.post_processors = ['truncate_line']
        else: args.post_processors = []
        logger.info(f"Using default post-processors for '{completion_type_for_pp}': {args.post_processors}")
    
    cache = {}
    if os.path.exists(args.cache_path):
        try:
            with open(args.cache_path, "r", encoding="utf-8") as f: cache = json.load(f)
            logger.info(f"Loaded cache with {len(cache)} entries from {args.cache_path}.")
        except Exception as e: logger.warning(f"Error loading cache from {args.cache_path}: {e}")

    logger.info(f"Building model: {args.model_name}")
    model_wrapper = build_model(args)

    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset_full = load_dataset(args.dataset_name) # Load full dataset first
    
    completion_type = get_completion_type_from_dataset(args.dataset_name)
    logger.info(f"Using completion type: {completion_type}")
    
    dataset_filtered = [s for s in dataset_full if s.get("task_id") in PROBLEMATIC_TASK_IDS]
    logger.info(f"Filtered to {len(dataset_filtered)} problematic samples for debugging out of {len(dataset_full)} total.")
    
    if not dataset_filtered:
        logger.error(f"No problematic samples (IDs: {PROBLEMATIC_TASK_IDS}) found in dataset '{args.dataset_name}'!")
        return

    results = []
    
    # Use tqdm for progress bar on the filtered dataset
    for sample_idx, sample in enumerate(tqdm(dataset_filtered, desc="Processing Problematic Samples")):
        # logger.info(f"\n{'#'*80}") # Header is now inside evaluate_with_adversarial_attack
        # logger.info(f"Processing sample {sample_idx + 1}/{len(dataset_filtered)}: {sample.get('task_id')}")
        # logger.info(f"{'#'*80}")
        
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

    if os.path.exists(args.cache_path): # Save cache only if it was loaded or potentially modified
        with open(args.cache_path, "w", encoding="utf-8") as f: 
            json.dump(cache, f, indent=2)
        logger.info(f"Cache saved to {args.cache_path} ({len(cache)} entries).")
    
    with open(args.output_path, "w", encoding="utf-8") as f:
        for r_item in results: 
            f.write(json.dumps(r_item) + "\n")
    logger.info(f"Debug results for {len(results)} samples saved to: {args.output_path}")

    logger.info(f"\n{'='*80}\nSUMMARY OF {len(results)} PROCESSED PROBLEMATIC SAMPLES\n{'='*80}")
    
    critical_issues_found = 0
    for r in results:
        task_id = r.get('task_id', 'N/A')
        original_passed = r.get('original_passed', False)
        adv_found = r.get('adversarial_found', False)
        is_vulnerable = r.get('adversarial_status') == "VULNERABLE"
        
        if is_vulnerable and r.get("completion") == r.get("adversarial_completion"):
            critical_issues_found += 1
            logger.error(f"\nCRITICAL ISSUE CONFIRMED for Task ID: {task_id}")
            logger.error(f"  Original Passed: {original_passed}")
            logger.error(f"  Adversarial Found (Vulnerable): {adv_found}")
            logger.error(f"  Completions: IDENTICAL")
            logger.error(f"  Original Status: {r.get('original_result_status')}, Tests: {r.get('original_tests_passed')}/{r.get('original_tests_total')}")
            adv_details = r.get('adversarial_test_details', {})
            logger.error(f"  Adversarial Status: {adv_details.get('status')}, Tests: {adv_details.get('tests_passed')}/{adv_details.get('tests_total')}")
            logger.error(f"  This sample exhibits the 'identical completion, different outcome' problem.")
        elif is_vulnerable:
            logger.warning(f"\nTask ID: {task_id} - VULNERABLE (completions differ or not checked for equality here)")
            logger.warning(f"  Original Passed: {original_passed}, Adversarial Status: {adv_details.get('status')}")
        else:
            logger.info(f"\nTask ID: {task_id} - Not VULNERABLE or attack failed/skipped.")
            logger.info(f"  Original Passed: {original_passed}, Adversarial Status: {r.get('adversarial_status')}")
            logger.info(f"  Adversarial Failure Reason: {r.get('adversarial_failure_reason', 'N/A')}")


    if critical_issues_found > 0:
        logger.error(f"\nTotal CRITICAL ISSUES (identical completion, different outcome): {critical_issues_found}/{len(results)}")
    else:
        logger.info(f"\nNo critical issues (identical completion, different outcome) were flagged in this run for the {len(results)} processed samples.")

    logger.info(f"\nDetailed DEBUG log file saved to: {LOG_FILE_PATH_LESS}")
    logger.info("Review the console output for INFO/WARNING/ERROR messages and the log file for detailed DEBUG traces.")

if __name__ == "__main__":
    main()