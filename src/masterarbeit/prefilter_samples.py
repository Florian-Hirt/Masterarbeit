#!/usr/bin/env python
import json
import os
from pathlib import Path
import argparse
from code_completion_generation import incomplete_code_transform

def prefilter_samples(input_path, output_path):
    """
    Filter the dataset to include only samples that can be successfully transformed
    with the incomplete_code_transform function.
    """
    total_samples = 0
    successful = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            total_samples += 1
            try:
                obj = json.loads(line)
                original_code = obj.get("source_code", "")
                
                # Try to transform and only keep samples that can be transformed
                transformed_code = incomplete_code_transform(original_code)
                if transformed_code is None:
                    continue
                    
                # Add the incomplete code to the object
                obj["incomplete_code"] = transformed_code
                json.dump(obj, fout)
                fout.write("\n")
                successful += 1
                
            except Exception as e:
                print(f"Error processing sample {total_samples}: {e}")
                continue
    
    print(f"Total samples processed: {total_samples}")
    print(f"Successfully transformed samples: {successful}")
    print(f"Filtered out: {total_samples - successful}")
    
    return successful

def main():
    parser = argparse.ArgumentParser(
        description="Filter dataset to include only samples that can be transformed."
    )
    parser.add_argument(
        "--input", type=str,
        default="data/code_summarization_data_python.jsonl",
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output", type=str,
        default="data/filtered_for_adversarial.jsonl",
        help="Path to the output filtered JSONL file."
    )
    
    args = parser.parse_args()
    prefilter_samples(args.input, args.output)

if __name__ == "__main__":
    main()