#!/usr/bin/env python
import json
import sys
from pathlib import Path
import argparse
import re


from .perturbation import perturbation
import lib2to3.refactor

FIXERS = lib2to3.refactor.get_fixers_from_package("lib2to3.fixes")
refactor_tool = lib2to3.refactor.RefactoringTool(FIXERS)

def preprocess_code(code: str) -> str:
    code = re.sub(r'\u00A0', ' ', code)
    
    code = re.sub(r'(?m)^\s*print\s+([^(\n]+)$', r'print(\1)', code)
    code = re.sub(r'(?m)^\s*print\s+([^(\n]+),\s*$', r'print(\1, end=" ")', code)
    return code


def convert_to_py3(code: str) -> str:
    try:
        new_code = str(refactor_tool.refactor_string(code, ""))
        return new_code
    except Exception as e:
        print(f"Conversion error: {e}")
        preprocessed = preprocess_code(code)
        try:
            new_code = str(refactor_tool.refactor_string(preprocessed, ""))
            return new_code
        except Exception as e2:
            print(f"Second conversion attempt failed: {e2}")
            return code  # Fallback: Originalcode zurückgeben
        

def process_file(input_path, output_path, unchanged_output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    unchanged_output_path = Path(unchanged_output_path)
    
    if not input_path.exists():
        print(f"Input file {input_path} does not exist!")
        sys.exit(1)
    
    total_count = 0
    direct_count = 0
    converted_count = 0
    failed_count = 0
    
    unchanged_samples = []
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            # For testing purposes, you can uncomment the following lines to limit the number of samples processed
            # if total_count == 6:
            #     break
            total_count += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")
                continue
            
            if "incomplete_code" in obj:
                original_code = obj["incomplete_code"]
                print(f"Processing sample {total_count}...")
                print(original_code)
                try:
                    perturbed_code, total_reorderings = perturbation(original_code)
                    direct_count += 1
                except Exception as e:
                    print(f"Direct perturbation failed: {e}")
                    try:
                        converted_code = convert_to_py3(original_code)
                        perturbed_code, total_reorderings = perturbation(converted_code)
                        print(total_reorderings)
                        converted_count += 1
                    except Exception as e2:
                        print(f"Converting and perturbation failed: {e2}")
                        perturbed_code = original_code
                        total_reorderings = None
                        failed_count += 1
                        obj["perturbation_error"] = str(e2)
                        unchanged_samples.append(obj)
                        continue
                obj["incomplete_code"] = perturbed_code
                obj["total_reorderings"] = total_reorderings
            else:
                print("Warning: JSON object does not contain a 'source_code' field.")
            
            fout.write(json.dumps(obj) + "\n")
    
    print(f"Overall samples: {total_count}")
    print(f"Directly perturbed: {direct_count}")
    print(f"Perturbed after conversion: {converted_count}")
    print(f"Unchanged (error): {failed_count}")
    print(f"Perturbation applied successfully. Output written to {output_path}")
    
    if unchanged_samples:
        with open(unchanged_output_path, 'w', encoding='utf-8') as f_un:
            for obj in unchanged_samples:
                minimal_obj = {
                    "incomplete_code": obj.get("incomplete_code", ""),
                    "perturbation_error": obj.get("perturbation_error", "")
                }
                f_un.write(json.dumps(minimal_obj) + "\n")
        print(f"{len(unchanged_samples)} unchanged samples saved to {unchanged_output_path}.")

def main():
    parser = argparse.ArgumentParser(
        description="Apply code perturbation to the 'source_code' field of a JSONL file."
    )
    parser.add_argument(
        "--input", type=str,
        default="data/incomplete_code_summarization_data_python.jsonl",
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output", type=str,
        default="data/incomplete_code_summarization_data_python_reorder_rename.jsonl",
        help="Path to the output (perturbed) JSONL file."
    )
    parser.add_argument(
        "--unchanged", type=str,
        default="data/unveraenderte_samples.jsonl",
        help="Path to the output file for unchanged samples."
    )
    
    args = parser.parse_args()
    process_file(args.input, args.output, args.unchanged)

if __name__ == "__main__":
    main()
