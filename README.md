# Clinical Text Structuring with Fine-Tuned LLM

## Overview
This project demonstrates a lightweight, reproducible pipeline for fine-tuning a language model to perform **structured text transformation** in domain-specific workflows.

The model converts **informal, shorthand-style inputs into clear, structured summaries**, simulating real-world documentation and communication pipelines.

---

## Problem Statement
In many operational environments, critical information is captured in:
- Abbreviated formats  
- Inconsistent phrasing  
- Hard-to-parse free text  

This creates downstream challenges for:
- Documentation quality  
- Communication between stakeholders  
- Data usability and analysis  

---

## Approach
- **Base Model**: FLAN-T5-base  
- **Fine-Tuning Method**: LoRA (parameter-efficient adaptation)  
- **Dataset**: Synthetic dataset inspired by real-world shorthand patterns  
- **Framework**: Hugging Face Transformers  

Rather than scaling model size, this project focuses on:
> High-quality data design + efficient adaptation

---

## Task Definition
### Input → Output Transformation

**Input:**
Pt c/o SOB x3d, hx COPD, O2 sat 89% RA

**Output:**
Patient reports shortness of breath for 3 days with a history of COPD. Oxygen saturation is 89% on room air.

##**Project Structure**
      Follows the general project folder structure

 ##setup
 #Install dependencies: pip install -r requirements.txt

 ##Training
 Run training with:
   python src/train.py --epochs 5

This will:
- Load and preprocess the dataset  
- Apply LoRA to the base model  
- Fine-tune on the transformation task  
- Save the model to `fine_tuned_model/`  

---

## Inference

Generate predictions using:
python src/inference.py

Outputs are saved to: outputs/sample_predictions.txt

---

## Key Design Decisions

### 1. Parameter-Efficient Fine-Tuning
Used LoRA to:
- Reduce compute requirements  
- Enable training on limited hardware  
- Maintain base model generalization  

---

### 2. Synthetic Data Strategy
Instead of relying on sensitive or proprietary data:
- Created structured synthetic examples  
- Focused on high-signal patterns (abbreviations, structure)  

---

### 3. Problem Framing
This project treats LLMs as:
> **Transformation engines within a workflow**, not general-purpose chat systems

---

##  Evaluation Approach
Evaluation is primarily qualitative, focusing on:
- Clinical completeness  
- Semantic accuracy  
- Consistency of structure  

Future improvements could include:
- Automated scoring metrics  
- Human-in-the-loop validation  
- Domain-specific evaluation benchmarks  

---

##  Data & Privacy
All data used in this project is:
- Synthetic  
- Publicly inspired  

No real-world or sensitive data is included. Dataa is synthetic

---

##  Future Work

- Expand dataset diversity and edge cases  
- Add evaluation metrics and benchmarks  
- Introduce structured output formats (JSON, templates)  
- Build a lightweight API or demo interface  
- Integrate into a larger document processing pipeline  

---

##  Key Takeaways

- Data quality often matters more than model size  
- Efficient fine-tuning enables practical domain adaptation  
- Framing the right problem is critical for real-world impact  

---


 
 


### Input → Output Transformation

**Input:**
