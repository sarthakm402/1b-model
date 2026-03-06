from sklearn.model_selection import train_test_split
import pandas as pd
import json
import os
file_path=r"/home/sarthak/Desktop/work/ml_code/1b-model/training_data_FULL_500.txt"
with open(file_path,"r",encoding="utf-8") as f:
 content=f.read()
blocks = content.split("Question:")
dataset = []
for block in blocks[1:]:
    if "Reasoning:" in block and "Answer:" in block:
        question, rest = block.split("Reasoning:")
        reasoning, answer = rest.split("Answer:")
        formatted_text = (
            "<|user|>\n"
            + question.strip()
            + "\n\n<|assistant|>\nReasoning:\n"
            + reasoning.strip()
            + "\n\nAnswer:\n"
            + answer.strip()
        )

        dataset.append({"text": formatted_text})
 
 
print("Total samples:", len(dataset))
train, val = train_test_split(dataset, test_size=0.2, random_state=42,shuffle=True)
with open("train.jsonl", "w") as f:
    for sample in train:
        f.write(json.dumps(sample) + "\n")
with open("val.jsonl", "w") as f:
    for sample in val:
        f.write(json.dumps(sample) + "\n")