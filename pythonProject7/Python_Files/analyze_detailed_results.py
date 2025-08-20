# analyze_detailed_results.py

import argparse
import json
from collections import Counter, defaultdict
import os

def stream_jsonl(filename: str):
    with open(filename, "r", encoding="utf-8") as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                yield json.loads(line)

def main():
    parser = argparse.ArgumentParser(description="Analyze detailed SAFIM evaluation results with test counts.")
    parser.add_argument("results_path", type=str, help="Path to the results_adversarial.jsonl file")
    args = parser.parse_args()

    if not os.path.exists(args.results_path):
        print(f"Error: Results file not found at {args.results_path}")
        return

    all_results = list(stream_jsonl(args.results_path))
    total_samples = len(all_results)
    print(f"Analyzing {total_samples} samples from {args.results_path}\n")

    evaluation_errors = sum(1 for r in all_results if r.get("error_in_evaluation"))
    print(f"Samples with evaluation errors: {evaluation_errors}")
    
    python_samples_with_tests = [
        r for r in all_results 
        if r.get("lang") == "python" and 
           r.get("original_tests_total", 0) > 0 and 
           not r.get("error_in_evaluation")
    ]
    print(f"Python samples with unit tests (and no eval error): {len(python_samples_with_tests)}")

    if not python_samples_with_tests:
        print("No Python samples with test details to analyze.")
        return

    # --- Analysis of Original Completions ---
    print("\n--- Original Completion Test Performance (Python Samples with Tests) ---")
    original_passed_all_tests = 0
    original_failed_some_tests = 0
    original_failed_distribution = Counter() # counts how many samples failed X tests
    
    total_original_unit_tests = 0
    total_original_unit_tests_passed = 0

    for r in python_samples_with_tests:
        total_original_unit_tests += r.get("original_tests_total", 0)
        total_original_unit_tests_passed += r.get("original_tests_passed", 0)
        
        if r.get("original_passed"):
            original_passed_all_tests += 1
        else:
            original_failed_some_tests += 1
            failed_count = r.get("original_tests_failed", 0)
            original_failed_distribution[failed_count] += 1
            
    print(f"  Samples where original completion passed all tests: {original_passed_all_tests}")
    print(f"  Samples where original completion failed at least one test: {original_failed_some_tests}")
    if total_original_unit_tests > 0:
        pass_rate = (total_original_unit_tests_passed / total_original_unit_tests) * 100
        print(f"  Overall original unit test pass rate: {total_original_unit_tests_passed}/{total_original_unit_tests} ({pass_rate:.2f}%)")
    if original_failed_some_tests > 0:
        print("  Distribution of original failures (number of tests failed : count of samples):")
        for k, v in sorted(original_failed_distribution.items()):
            print(f"    Failed {k} tests: {v} samples")

    # --- Analysis of Adversarial Attacks ---
    print("\n--- Adversarial Attack Test Performance (on Vulnerable Python Samples) ---")
    vulnerable_samples = [
        r for r in python_samples_with_tests 
        if r.get("original_passed") and r.get("adversarial_found")
    ]
    print(f"Vulnerable samples (original passed, adversarial failed): {len(vulnerable_samples)}")

    if vulnerable_samples:
        adv_failed_distribution = Counter()
        total_adv_unit_tests_on_vulnerable = 0
        total_adv_unit_tests_failed_on_vulnerable = 0

        for r in vulnerable_samples:
            details = r.get("adversarial_test_details")
            if details:
                total_adv_unit_tests_on_vulnerable += details.get("tests_total", 0)
                failed_count = details.get("tests_failed", 0)
                total_adv_unit_tests_failed_on_vulnerable += failed_count
                adv_failed_distribution[failed_count] += 1
        
        if total_adv_unit_tests_on_vulnerable > 0:
            adv_fail_rate = (total_adv_unit_tests_failed_on_vulnerable / total_adv_unit_tests_on_vulnerable) * 100
            print(f"  For vulnerable samples, overall adversarial code unit test fail rate: {total_adv_unit_tests_failed_on_vulnerable}/{total_adv_unit_tests_on_vulnerable} ({adv_fail_rate:.2f}%)")
        print("  Distribution of adversarial failures (number of tests failed by adv. code : count of samples):")
        for k, v in sorted(adv_failed_distribution.items()):
            print(f"    Adv. code failed {k} tests: {v} samples")
    
    # List samples that failed many tests (example)
    # threshold = 3 
    # print(f"\n--- Samples where original code failed >= {threshold} tests ---")
    # for r in python_samples_with_tests:
    #     if not r.get("original_passed") and r.get("original_tests_failed", 0) >= threshold:
    #         print(f"  Task ID: {r['task_id']}, Failed Tests: {r['original_tests_failed']}/{r['original_tests_total']}")


if __name__ == "__main__":
    main()