import json
import numpy as np

input_file = "/nfs/homedirs/hifl/Masterarbeit/pythonProject7/Python_Files/results_gemma-3-4b-it/filtered_adversarial_data.jsonl"

with open(input_file, 'r') as fin:
    human_judgements = []
    adversarial_scores = []
    for line in fin:
        # Parse the JSON object
        data = json.loads(line)
        
        # Extract the human summarization
        human_judgement = data.get("human_rating", "")
        adversarial_score = data.get("adversarial_score", "")
        
        # Append to the list
        human_judgements.append(human_judgement)
        adversarial_scores.append(adversarial_score)
    
    print("Human judgement: ", human_judgements)
    print("Adversarial socres: ", adversarial_scores)
    correlation = np.corrcoef(human_judgements, adversarial_scores)[0, 1]

print(f"Correlation between human judgement and adversarial score: {correlation}")