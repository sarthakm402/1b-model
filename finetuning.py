import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Tra iner
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from sentence_transformers import SentenceTransformer, util

train_path = "/home/sarthak/Desktop/work/ml_code/1b-model/train.jsonl"
val_path = "/home/sarthak/Desktop/work/ml_code/1b-model/val.jsonl"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

train_dataset = load_dataset("json", data_files={"train": train_path})["train"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

base_model = prepare_model_for_kbit_training(base_model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)

def tokenize(sample):
    tokens = tokenizer(sample["text"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_dataset = train_dataset.map(tokenize, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir="lora_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

model.save_pretrained("lora_adapter")
tokenizer.save_pretrained("lora_adapter")

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, "lora_adapter")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_text(sample):
    text = sample["text"]
    if "<|assistant|>" in text:
        return text.split("<|assistant|>")[0]
    return text

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False, temperature=0)
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

exact_correct = 0
semantic_correct = 0
total = 0

with open(val_path, "r") as f:
    for line in f:
        sample = json.loads(line)
        prompt = get_text(sample)
        output = generate(prompt)
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
            else:
                print("QUESTION:", prompt)
                print("EXPECTED:", expected)
                print("PREDICTED:", predicted)
                print("SIMILARITY:", score)
                print("-"*40)
        total += 1

exact_accuracy = exact_correct / total
semantic_accuracy = semantic_correct / total

print("Exact Accuracy:", exact_accuracy)
print("Semantic Accuracy:", semantic_accuracy)
