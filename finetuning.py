import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import accelerate
import datasets
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model,PeftModel
from transformers import TrainingArguments
from transformers import Trainer
train_dataset=datasets.load_dataset("json",data_files={"train": r"/home/sarthak/Desktop/work/ml_code/1b-model/train.jsonl"})
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)
base_model_before = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model_before, lora_config)
def tokenize(sample):
    return tokenizer(
        sample["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

train_dataset = train_dataset.map(tokenize)
training_args = TrainingArguments(
    output_dir="lora_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
model.save_pretrained("lora_adapter")
tokenizer.save_pretrained("lora_adapter")
base_model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, "lora_adapter")