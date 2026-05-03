import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model
@st.cache_resource
def load_model():
    model_path = "fine_tuned_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# UI
st.title("Clinical Text Structuring LLM")
st.write("Convert shorthand clinical notes into structured summaries.")

user_input = st.text_area(
    "Enter shorthand clinical note:",
    "Pt c/o SOB x3d, hx COPD, O2 sat 89% RA"
)

def generate(text):
    prompt = f"Convert the shorthand clinical note into a structured summary.\nInput: {text}\nOutput:"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if st.button("Generate Structured Output"):
    with st.spinner("Generating..."):
        result = generate(user_input)
        st.subheader("Structured Output")
        st.write(result)
