import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util

base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
lora_path = "lora_adapter"
val_path = "val.jsonl"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to("cuda")
lora_base = AutoModelForCausalLM.from_pretrained(base_model_name).to("cuda")
lora_model = PeftModel.from_pretrained(lora_base, lora_path)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_text(sample):
    text = sample["text"]
    if "<|assistant|>" in text:
        return text.split("<|assistant|>")[0]
    return text

def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        temperature=0
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

def extract_answer(text):
    match = re.search(r"answer\s*:\s*(.*)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

def normalize(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def contains_match(pred, gold):
    return normalize(gold) in normalize(pred)

def semantic_match(pred, gold, threshold=0.60):
    emb1 = embed_model.encode(pred, convert_to_tensor=True)
    emb2 = embed_model.encode(gold, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    return score >= threshold, score

def evaluate(model, name):

    exact_correct = 0
    semantic_correct = 0
    total = 0

    with open(val_path, "r") as f:
        for line in f:
            sample = json.loads(line)

            prompt = get_text(sample)
            output = generate(model, prompt)

            predicted = extract_answer(output)
            expected = extract_answer(sample["text"])

            pred = normalize(predicted)
            gold = normalize(expected)

            if pred == gold or contains_match(predicted, expected):
                exact_correct += 1
                semantic_correct += 1
            else:
                sem_ok, score = semantic_match(predicted, expected)
                if sem_ok:
                    semantic_correct += 1

            total += 1

    exact_accuracy = exact_correct / total
    semantic_accuracy = semantic_correct / total

    print("MODEL:", name)
    print("Exact Accuracy:", exact_accuracy)
    print("Semantic Accuracy:", semantic_accuracy)
    print("-" * 40)

evaluate(base_model, "BASE MODEL")
evaluate(lora_model, "LORA MODEL")