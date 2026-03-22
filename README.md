# 🚀 1B ML + Code Reasoning Model (Under 4GB VRAM)

A lightweight, efficient fine-tuning pipeline for building a **code + ML reasoning model** that runs under **4GB VRAM**, using **LoRA-based parameter-efficient training** and a structured evaluation harness.

This project focuses on making **small models actually useful**, not just training large ones.

---

## ⚠️ What This Project Really Is

- Not a foundation model  
- Not state-of-the-art research  
- A practical system for:
  - Fine-tuning small models
  - Improving reasoning on ML/code tasks
  - Evaluating improvements properly  

---

## 🧠 Core Idea

Instead of brute-force training:

- Use **LoRA (Low-Rank Adaptation)** to fine-tune efficiently  
- Keep VRAM usage minimal  
- Build an **evaluation harness** to measure real improvement  
- Compare against baseline models  

---

## 📁 Project Structure


.
├── lora_adapter/
├── lora_model/
├── before_model.py
├── finetuning.py
├── comparison.py
├── split.py
├── train.jsonl
├── val.jsonl
├── training_data_FULL_500.txt
├── 1B ML + Code Reasoning Model under 4GB VRAM.odt


---

## 🔧 Key Components

### 1. Fine-Tuning Pipeline (`finetuning.py`)

- Implements LoRA-based fine-tuning  
- Optimized for low VRAM (~4GB)  
- Uses structured reasoning-focused data  

---

### 2. LoRA Adapter (`lora_adapter/`)

**LoRA (Low-Rank Adaptation):**

- Freeze original model weights  
- Train small low-rank matrices  
- Inject into transformer layers  

**Benefits:**
- Lower memory usage  
- Faster training  
- Efficient adaptation  

---

### 3. Evaluation Harness (`comparison.py`)

- Baseline vs fine-tuned comparison  
- Structured evaluation pipeline  
- Output quality tracking  

---

### 4. Dataset Pipeline

- `training_data_FULL_500.txt` → raw dataset  
- `split.py` → generates:
  - `train.jsonl`
  - `val.jsonl`

---

### 5. Baseline Model (`before_model.py`)

- Used for comparison  
- Prevents misleading evaluation  

---

## ⚙️ Setup & Usage

### 1. Install Dependencies

```bash
pip install torch transformers datasets peft accelerate
2. Prepare Data
python split.py
3. Fine-Tune Model
python finetuning.py
4. Evaluate Performance
python comparison.py
📊 Expected Outcomes
Improved ML and code reasoning
Better structured outputs
Slight reduction in hallucinations

If results are poor:

Dataset quality is likely weak
Prompts may be shallow
Evaluation may be insufficient
