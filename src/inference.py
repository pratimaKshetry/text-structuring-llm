from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "fine_tuned_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def generate(text):
    prompt = f"Convert the shorthand clinical note into a structured summary.\nInput: {text}\nOutput:"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    test_inputs = [
        "Pt c/o SOB x2d, hx asthma, O2 92%",
        "60M w/ CP, BP 160/100, denies dizziness"
    ]

    with open("outputs/sample_predictions.txt", "w") as f:
        for inp in test_inputs:
            result = generate(inp)
            f.write(f"Input: {inp}\nOutput: {result}\n\n")
            print(f"Input: {inp}")
            print(f"Output: {result}\n")
