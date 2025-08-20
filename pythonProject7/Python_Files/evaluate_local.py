import argparse
import ast
import json
import os
import re
import subprocess
import tempfile
import logging
from tqdm import tqdm

# From safim library
from safim.ast_utils import ErrorCheckVisitor, get_parser
from safim.data_utils import load_dataset, stream_jsonl
# We will inline/adapt syntax_match and parts of run_test logic

# --- Logging Setup ---
LOG_FILE_PATH = "evaluation_log.txt"
if os.path.exists(LOG_FILE_PATH):
    os.remove(LOG_FILE_PATH)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
# --- End of Logging Setup ---

def check_python_syntax(code_string: str) -> bool:
    """Uses safim's AST checker for Python syntax."""
    parser = get_parser("python")
    code_bytes = code_string.encode("utf-8")
    tree = parser.parse(code_bytes)
    error_check = ErrorCheckVisitor()
    error_check(tree)
    return error_check.error_cnt == 0

def get_function_call_params(node): # From safim.evaluate
    positional_args = [ast.dump(arg) for arg in node.args]
    keyword_args = {kw.arg: ast.dump(kw.value) for kw in node.keywords}
    return positional_args, keyword_args

def function_calls_match(call1, call2): # From safim.evaluate
    params1 = get_function_call_params(call1)
    params2 = get_function_call_params(call2)
    return params1 == params2

def syntax_match(code1: str, code2: str, lang: str) -> bool: # Adapted from safim.evaluate
    """Checks if two code snippets are syntactically equivalent or string equivalent."""
    # Normalize whitespace for a basic check
    norm_code1 = re.sub(r'\s+', '', code1).strip()
    norm_code2 = re.sub(r'\s+', '', code2).strip()

    if lang == "python":
        try:
            # For API calls, they might be expressions, try 'eval' mode
            tree1 = ast.parse(norm_code1, mode='eval')
            tree2 = ast.parse(norm_code2, mode='eval')
            if isinstance(tree1.body, ast.Call) and isinstance(tree2.body, ast.Call):
                if function_calls_match(tree1.body, tree2.body):
                    return True
        except SyntaxError:
            # If 'eval' fails, try 'exec' mode for statements/blocks
            try:
                tree1_exec = ast.parse(code1.strip(), mode='exec')
                tree2_exec = ast.parse(code2.strip(), mode='exec')
                # Basic check: compare AST dumps (ignoring minor formatting)
                if ast.dump(tree1_exec) == ast.dump(tree2_exec):
                    return True
            except SyntaxError:
                pass # Fall through to string comparison if AST parsing fails completely
        except Exception as e:
            logger.debug(f"AST comparison failed for '{lang}': {e}")
            pass


    # Fallback to normalized string comparison
    return norm_code1 == norm_code2


def local_python_test_runner(code_string: str, unit_tests: list) -> tuple[str, bool]:
    """
    Runs Python code against a list of unit tests locally.
    Returns (result_status, passed_all_tests).
    """
    if not unit_tests:
        return "NO_TESTS", True # Or False if no tests means failure

    # First, check overall syntax of the generated code block
    # The eval_prompt in safim usually provides context to make the completion a full script
    if not check_python_syntax(code_string):
        logger.warning(f"Syntax error in full code block:\n{code_string[:500]}...")
        return "COMPILATION_ERROR", False

    all_tests_passed = True
    overall_result_status = "PASSED" # Assume pass until a test fails

    for i, test_case in enumerate(unit_tests):
        test_input = test_case.get("input", "")
        expected_outputs = test_case.get("output", []) # list of strings

        temp_file_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp_code_file:
                tmp_code_file.write(code_string)
                temp_file_path = tmp_code_file.name
            
            # For unit tests, often the test itself is an assertion within the problem's "eval_prompt"
            # If unit_tests has "input" and "output", it implies we feed "input" and check "output".
            # Safim's `problem['unit_tests']` seems to be structured as list of dicts with 'input' and 'output' keys.
            # The `problem['eval_prompt']` should contain the necessary boilerplate and the `{{completion}}`.
            # The 'input' for the unit test should be piped to stdin.
            # The 'output' should be compared against stdout.

            logger.debug(f"Executing test {i+1} with input: '{test_input[:100]}'")
            process = subprocess.run(
                ["python", temp_file_path],
                input=test_input,
                text=True,
                capture_output=True,
                timeout=10  # Seconds
            )

            if process.returncode != 0:
                logger.warning(f"Test {i+1} RUNTIME_ERROR. Stderr: {process.stderr[:500]}")
                all_tests_passed = False
                overall_result_status = "RUNTIME_ERROR"
                break 

            actual_stdout_lines = process.stdout.strip().splitlines()
            # Normalize expected outputs: strip whitespace from each line
            normalized_expected_outputs = [line.strip() for line in expected_outputs]
            normalized_actual_outputs = [line.strip() for line in actual_stdout_lines]


            if normalized_actual_outputs != normalized_expected_outputs:
                logger.warning(f"Test {i+1} WRONG_ANSWER.")
                logger.warning(f"Expected: {normalized_expected_outputs}")
                logger.warning(f"Actual:   {normalized_actual_outputs}")
                all_tests_passed = False
                overall_result_status = "WRONG_ANSWER"
                break
            logger.debug(f"Test {i+1} passed.")

        except subprocess.TimeoutExpired:
            logger.warning(f"Test {i+1} TIMEOUT.")
            all_tests_passed = False
            overall_result_status = "TIME_LIMIT_EXCEEDED"
            break
        except Exception as e:
            logger.error(f"Error during test {i+1} execution: {e}", exc_info=True)
            all_tests_passed = False
            overall_result_status = "RUNTIME_ERROR" # Or a custom "EXECUTION_FRAMEWORK_ERROR"
            break
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    return overall_result_status, all_tests_passed


def main():
    parser = argparse.ArgumentParser(description="Local SAFIM Evaluation Script (Python focused)")
    parser.add_argument("--completion_type", default="control", type=str, help="Task type (e.g., 'api', 'block')")
    parser.add_argument("--completion_path", default="outputs_control/deepseek-coder-1.3b-python_only.jsonl", type=str, help="Path to the .jsonl file with generated completions.")
    parser.add_argument("--output_path", default="/nfs/homedirs/hifl/Masterarbeit/pythonProject7/Python_Files/evaluations/evaluation1.jsonl", type=str, help="Path to save evaluation results.")
    # Add any other arguments from safim/evaluate.py if needed, like --port (though we won't use it)
    args = parser.parse_args()

    logger.info(f"Starting local evaluation with args: {args}")

    completions_data = {comp["task_id"]: comp for comp in stream_jsonl(args.completion_path)}
    logger.info(f"Loaded {len(completions_data)} completions from {args.completion_path}")

    pass_count = 0
    total_count = 0
    results_summary = []

    dataset = load_dataset(args.completion_type)
    logger.info(f"Loaded {len(dataset)} problems for task type '{args.completion_type}'")

    for problem in tqdm(dataset, desc="Evaluating Problems"):
        task_id = problem["task_id"]
        lang = problem["lang"]
        
        # We are focusing on Python for local execution
        if lang != "python":
            # Optionally, still try syntax_match for other languages if it makes sense
            # Or just skip non-Python if you only want to evaluate Python exec
            logger.debug(f"Skipping non-Python problem: {task_id} (lang: {lang})")
            # result_entry = {"task_id": task_id, "result": "SKIPPED_NON_PYTHON", "passed": False}
            # results_summary.append(result_entry)
            # total_count +=1 # Decide if skipped non-python count towards total
            continue

        if task_id not in completions_data:
            logger.warning(f"Completion for {task_id} not found in {args.completion_path}. Skipping its evaluation.")
            # Optionally, you might want a separate counter for missing completions if that's useful
            # For Pass@k, these should not be in the denominator of successfully attempted problems
            results_summary.append({ # Still log that it was missing
                "task_id": task_id,
                "lang": lang,
                "result": "MISSING_COMPLETION",
                "passed": False,
                "completion_used": "N/A (Missing)"
            })
            continue # Skip to the next problem


        total_count += 1
        result_status = "UNKNOWN_FAILURE" # Default status
        passed = False

        if task_id not in completions_data:
            logger.warning(f"Completion for {task_id} not found in {args.completion_path}")
            result_status = "MISSING_COMPLETION"
        else:
            completion_entry = completions_data[task_id]
            generated_completion = completion_entry.get("completion")

            if generated_completion is None or not generated_completion.strip():
                result_status = "EMPTY_COMPLETION"
            else:
                logger.info(f"\n--- Evaluating Task ID: {task_id} ---")
                logger.info(f"Language: {lang}")
                logger.info(f"Generated Completion (snippet): {generated_completion[:100].replace(os.linesep, ' ')}...")
                logger.info(f"Ground Truth (snippet): {problem['ground_truth'][:100].replace(os.linesep, ' ')}...")

                if "unit_tests" in problem and problem["unit_tests"]:
                    logger.info(f"Found {len(problem['unit_tests'])} unit tests.")
                    full_code_to_execute = problem['eval_prompt'].replace("{{completion}}", generated_completion)
                    
                    # Initial syntax check of the whole runnable code
                    if not check_python_syntax(full_code_to_execute):
                        result_status = "COMPILATION_ERROR"
                        passed = False
                        logger.warning(f"Task {task_id}: Overall COMPILATION_ERROR before running tests.")
                    else:
                        # If overall syntax is okay, try running unit tests
                        if generated_completion.strip() == problem["ground_truth"].strip():
                             result_status = "PASSED_GT_MATCH" # Exact match to GT, assume tests pass if GT is correct
                             passed = True
                             logger.info(f"Task {task_id}: Exact match to ground truth. Marked as PASSED_GT_MATCH.")
                        else:
                            result_status, passed = local_python_test_runner(full_code_to_execute, problem["unit_tests"])
                            logger.info(f"Task {task_id}: Unit test run result: {result_status}, Passed: {passed}")

                else: # No unit tests, rely on syntax_match
                    logger.info("No unit tests found, using syntax_match.")
                    if syntax_match(generated_completion, problem["ground_truth"], lang):
                        result_status = "EXACT_SYNTAX_MATCH"
                        passed = True
                    else:
                        result_status = "WRONG_ANSWER_SYNTAX_MISMATCH"
                        passed = False
                    logger.info(f"Task {task_id}: Syntax match result: {result_status}, Passed: {passed}")
        
        if passed:
            pass_count += 1
        
        results_summary.append({
            "task_id": task_id,
            "lang": lang,
            "result": result_status,
            "passed": passed,
            "completion_used": completions_data.get(task_id, {}).get("completion", "N/A") if task_id in completions_data else "N/A (Missing)"
        })
        logger.info(f"--- Finished Task ID: {task_id}, Result: {result_status}, Passed: {passed} ---")


    logger.info("\n--- Evaluation Summary ---")
    if total_count > 0:
        pass_rate = (pass_count / total_count) * 100
        logger.info(f"Total Python Problems Evaluated: {total_count}")
        logger.info(f"Passed: {pass_count}")
        logger.info(f"Pass Rate (Pass@1 for Python): {pass_rate:.2f}%")
    else:
        logger.info("No Python problems were evaluated.")

    # Save detailed results
    with open(args.output_path, "w", encoding="utf-8") as f:
        for res in results_summary:
            f.write(json.dumps(res) + "\n")
    logger.info(f"Detailed evaluation results saved to {args.output_path}")
    logger.info(f"Evaluation log saved to {LOG_FILE_PATH}")
    logger.info("Evaluation script finished.")

if __name__ == '__main__':
    main()