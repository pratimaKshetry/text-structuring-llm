import json
import argparse
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model


def load_data(path):
    with open(path) as f:
        data = json.load(f)
    return Dataset.from_list(data)


def preprocess(example, tokenizer):
    prompt = f"{example['instruction']}\nInput: {example['input']}\nOutput:"

    inputs = tokenizer(
        prompt,
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        example["output"],
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    inputs["labels"] = labels["input_ids"]
    return inputs


def main(args):
    dataset = load_data(args.data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

    model = get_peft_model(model, lora_config)

    dataset = dataset.map(lambda x: preprocess(x, tokenizer))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="data/synthetic_dataset.json")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_model")
    parser.add_argument("--epochs", type=int, default=5)

    args = parser.parse_args()

    main(args)
