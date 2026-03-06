from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
val_data=r"/home/sarthak/Desktop/work/ml_code/1b-model/val.jsonl"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
def get_text(sample):
    text=sample["text"]
    question=text.split("<|assistant|>")[0]
    return question
def generate(prompt):
    inputs=tokenizer(prompt,return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
def extract_answer(text):

    if "Answer:" in text:
        return text.split("Answer:")[-1].strip()

    return text.strip()
correct = 0

for sample in val_data:

    prompt = get_text(sample)

    output = generate(prompt)

    predicted = extract_answer(output)

    expected = extract_answer(sample["text"])

    if predicted == expected:
        correct += 1

accuracy = correct / len(val_data)

print("Baseline accuracy:", accuracy)