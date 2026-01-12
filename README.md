# Marketing KPI Insights with Generative AI (LoRA Fine-Tuning)

**Version:** v1.0.0  

This is a portfolio project demonstrating the use of **Generative AI** to generate actionable marketing KPI insights from raw campaign metrics. The project leverages **LoRA fine-tuning** on a synthetic dataset of marketing campaigns to build a lightweight, task-specific language model.

---

## Project Goal

The goal of this project is to showcase an **AI/GenAI Engineer portfolio project** where a language model is fine-tuned to understand marketing KPIs (impressions, conversions, budgets, and conversion rates) and provide **actionable recommendations** for campaign optimization.

---

## Dataset

- Synthetic marketing campaign dataset (~3000 records) created for training and testing.  
- Each record contains:
  - `instruction`: Task description ("Analyze campaign performance and suggest improvements")  
  - `input`: Campaign metrics (impressions, conversions, budget)  
  - `output`: Conversion rate calculation and actionable suggestion  

The dataset is stored as a JSONL file for easy loading with Hugging Face `datasets`.

---

## Project Structure

- **Phase 1 – Fine-Tuning**
  - Notebook: `Marketing KPI GenAI Phase 1 – Fine-Tuning`
  - Goal: Fine-tune a base LLM using **LoRA adapters** on the synthetic marketing KPI dataset.
  - Output: Fine-tuned model saved with tokenizer at `/kaggle/working/marketing_kpi_lora`
  
- **Phase 2 – Inference & Evaluation**
  - Notebook: `Marketing KPI GenAI Phase 2 – Inference & Insight`
  - Goal: Use the fine-tuned model to generate KPI insights from new campaign metrics.
  - Demonstrates model performance and actionable recommendations generation.

---

## Notebooks

- **Phase 1 – Fine-Tuning:** [Kaggle Notebook Link](https://www.kaggle.com/code/omarmamdooh/fine-tuning-on-marketing-kpis)
- **Phase 2 – Inference & Evaluation:** [Kaggle Notebook Link](https://www.kaggle.com/code/omarmamdooh/marketing-kpi-genai-phase-2-inference-insight)

---

## Features

- Fine-tuning a **pre-trained LLM** using **LoRA adapters** for lightweight adaptation
- Generates actionable insights for marketing campaigns
- Demonstrates **end-to-end GenAI workflow**: synthetic data → fine-tuning → inference
  
---

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load fine-tuned LoRA model
model_path = "/path/to/marketing_kpi_lora"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Example input
input_text = "Impressions: 50000, Conversions: 150, Budget: $2000"
prompt = f"Analyze campaign performance and suggest improvements.\nInput: {input_text}"

# Generate insights
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Technologies & Tools
- Python 3.12
- Hugging Face Transformers & Datasets
- PyTorch
- LoRA (Low-Rank Adaptation)
- Kaggle Notebooks (for development and demo)
- JSONL for dataset storage

---

License

This project is released under the MIT License.
