# 🌐 Multilingual Sentiment Analysis with LLaMA 3.1 + LoRA

Fine-tuning **LLaMA 3.1 8B Instruct** with **QLoRA** for binary sentiment classification across multiple languages, using culturally-aware prompting.

## 📌 Overview

This project fine-tunes LLaMA 3.1 (8B) as a sequence classifier on multilingual text using parameter-efficient fine-tuning (PEFT/LoRA). A custom prompt instructs the model to act as a cultural linguist, accounting for local idioms and nuances before classifying sentiment as Positive or Negative.

## 🧠 Approach

- **Model:** LLaMA 3.1 8B Instruct (`AutoModelForSequenceClassification`)
- **Quantization:** 4-bit QLoRA via BitsAndBytes (NF4, bfloat16)
- **Fine-tuning:** LoRA on `q_proj` and `v_proj` — only ~13.6M of 7.5B parameters trained (0.18%)
- **Prompt strategy:** Custom culturally-aware prompt that passes the language name alongside each sentence
- **Evaluation metric:** Weighted F1 Score

## 📁 Project Structure

```
multilingual-sentiment-analysis/
├── code.ipynb        # Main training notebook
└── README.md
```

## 🛠️ Tech Stack

- Python, PyTorch
- HuggingFace Transformers, PEFT, Datasets
- BitsAndBytes (4-bit quantization)
- scikit-learn (F1 evaluation)
- Kaggle T4 GPU

## ⚙️ Key Hyperparameters

| Parameter | Value |
|---|---|
| LoRA rank (r) | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.1 |
| Target modules | q_proj, v_proj |
| Learning rate | 1e-4 |
| Epochs | 3 |
| Batch size | 4 (+ grad accum ×2) |
| Max sequence length | 512 |

## 📊 Results

| Split | F1 Score |
|-------|----------|
| Validation (best) | 0.9146 |
| **Kaggle Test** | **0.95** |
