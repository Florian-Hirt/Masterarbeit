import json

input_file = "/nfs/homedirs/hifl/Masterarbeit/pythonProject7/Python_Files/results_gemma-3-4b-it/adversarial_reorderings_gemma3-1B.jsonl"
output_file = "/nfs/homedirs/hifl/Masterarbeit/pythonProject7/Python_Files/results_gemma-3-4b-it/filtered_adversarial_data.jsonl"

with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
    for line in fin:
        # Parse the JSON object
        data = json.loads(line)
        
        # Extract only the requested fields
        filtered_data = {
            "source_code": data.get("source_code", ""),
            "human_summarization": data.get("human_summarization", ""),
            "adversarial_score": data.get("adversarial_score", None),
            "adversarial_completion": data.get("adversarial_completion", "")
        }
        
        # Write the filtered data to the output file
        fout.write(json.dumps(filtered_data) + '\n')

print(f"Filtered data has been written to {output_file}")