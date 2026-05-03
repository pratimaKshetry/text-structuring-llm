import random
import json

# -----------------------------
# synthetic clinical-style text generation
# -----------------------------

# Common symptoms seen in shorthand clinical notes
symptoms = [
    "SOB",  # shortness of breath
    "CP",   # chest pain
    "HA",   # headache
    "dizziness",
    "fatigue",
    "nausea"
]

# Common chronic conditions for history simulation
conditions = [
    "COPD",
    "asthma",
    "DM2",
    "HTN",
    "migraine"
]

# Example vitals / clinical measurements
vitals = [
    "BP 150/90",
    "O2 sat 89%",
    "HR 52",
    "Temp 101F"
]

# Templates simulate real-world shorthand documentation styles
# These introduce variability in structure and phrasing
templates = [
    "Pt c/o {symptom} x{days}d, hx {condition}, {vital}",
    "{age}{sex} w/ {symptom}, {vital}, hx {condition}",
    "Pt reports {symptom} for {days} days, denies worsening, {vital}"
]


def generate_example():
    """
    Generate a single synthetic training example.

    Returns:
        dict: instruction tuning format (input/output pair)
    """

    # Randomly sample clinical attributes to simulate variability
    symptom = random.choice(symptoms)
    condition = random.choice(conditions)
    vital = random.choice(vitals)

    # Demographic variability
    age = random.randint(18, 85)
    sex = random.choice(["M", "F"])
    days = random.randint(1, 7)

    # Construct shorthand-style input note
    input_text = random.choice(templates).format(
        symptom=symptom,
        condition=condition,
        vital=vital,
        age=age,
        sex=sex,
        days=days
    )

    # Structured output (normalized clinical summary)
    # This is what the model learns to generate
    output_text = (
        f"Patient presents with {symptom}. "
        f"History includes {condition}. "
        f"Current finding: {vital}."
    )

    return {
        # Instruction tuning format for LLM fine-tuning
        "instruction": "Convert shorthand clinical note into structured summary.",
        "input": input_text,
        "output": output_text
    }


def generate_dataset(n=2000, out_path="data/synthetic_dataset.json"):
    """
    Generate a full synthetic dataset.

    Args:
        n (int): number of examples to generate
        out_path (str): output JSON file path
    """

    # Generate dataset using list comprehension for efficiency
    data = [generate_example() for _ in range(n)]

    # Save dataset to disk in JSON format
    # This simulates a real dataset pipeline output
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[INFO] Generated {n} synthetic examples → {out_path}")


if __name__ == "__main__":
    # Default execution entrypoint for reproducibility
    generate_dataset()
