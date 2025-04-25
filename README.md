# Spam Detection with Traditional ML vs. Zero-Shot LLM

This project compares traditional machine learning models (Logistic Regression and Naive Bayes) with a zero-shot large language model (LLM) for spam detection. 

I built this project to help one of my online students understand AI better in cyber security using hands on building and comparison approach. Maybe others will find it useful to expand upon also.

## 📁 Project Structure

```
project/
├── data/
│   └── emails.csv
├── ml_models/
│   ├── naive_bayes.py
│   ├── logistic_regression.py
├── llm/
│   ├── llm_classifier.py
│   └── llm_evaluation.py
├── results/
│   ├── llm_predictions.csv
│   ├── llm_metrics.txt
│   ├── llm_confusion_matrix.png
│   └── llm_roc_curve.png
└── README.md
```

## 📊 Dataset

- **Source**: SMS Spam Collection dataset
- **Format**: CSV with columns `text` (message) and `label` (`ham` or `spam`)

## ⚙️ Setup

1. Create a virtual environment and install requirements:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your Groq API key:

```env
GROQ_API_KEY=your_key_here
```

## 🤖 ML Models

Run the traditional ML classifiers and generate evaluation metrics:

```bash
python ml_models/logistic_regression.py
python ml_models/naive_bayes.py
```

## 🧠 LLM Zero-Shot Classification

Run the LLM-based classification (Groq API using LLaMA):

```bash
python llm/llm_classifier.py
```

Then evaluate results:

```bash
python llm/llm_evaluation.py
```

## 📈 Results

LLM and ML model outputs are saved to the `results/` folder, including:

- Accuracy and classification report
- Confusion matrix (PNG)
- ROC curve (PNG)

## 🧪 Next Steps

- Build a Gradio or Streamlit UI for interactive email classification
- Expand dataset and analyze performance across models

---

© 2025 Spam Classifier Project — Educational Use Only