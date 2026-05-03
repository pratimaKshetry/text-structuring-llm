import json
from datasets import Dataset

def load_dataset(path="data/synthetic_dataset.json"):
    with open(path) as f:
        data = json.load(f)
    return Dataset.from_list(data)

def format_prompt(example):
    return f"{example['instruction']}\nInput: {example['input']}\nOutput:"
