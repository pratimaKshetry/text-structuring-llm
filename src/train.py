import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

# Load data
with open("data/synthetic_dataset.json") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

# Model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_config)

# Preprocess
def preprocess(example):
    prompt = f"{example['instruction']}\nInput: {example['input']}\nOutput:"
    
    inputs = tokenizer(prompt, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(example["output"], max_length=256, truncation=True, padding="max_length")

    inputs["labels"] = labels["input_ids"]
    return inputs

dataset = dataset.map(preprocess)

# Training
training_args = TrainingArguments(
    output_dir="results",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    logging_steps=10,
    save_strategy="no",
    learning_rate=2e-4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

# Save
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
