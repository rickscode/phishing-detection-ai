import os
import gradio as gr
import pickle
from dotenv import load_dotenv
from groq import Groq
import numpy as np

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load ML Models and Vectorizers
with open("logistic_regression_model.pkl", "rb") as f:
    logistic_model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    logistic_vectorizer = pickle.load(f)

with open("naives_bayes_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

with open("nb_tfidf_vectorizer.pkl", "rb") as f:
    nb_vectorizer = pickle.load(f)

# Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)
LLM_MODEL = "llama3-8b-8192"

# Prompt for LLM

def build_prompt(email_text):
    return f"""
You are a cybersecurity assistant. Please classify the following email as either 'Spam' or 'Ham' (not spam).

Email:
"""
{email_text}
"""

Your answer must be just one word: Spam or Ham.
"""

# Prediction Functions

def predict_logistic(text):
    X = logistic_vectorizer.transform([text])
    pred = logistic_model.predict(X)[0]
    return "Spam" if pred == 1 else "Ham"

def predict_nb(text):
    X = nb_vectorizer.transform([text])
    pred = nb_model.predict(X)[0]
    return "Spam" if pred == 1 else "Ham"

def predict_llm(text):
    chat_completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for email spam detection."},
            {"role": "user", "content": build_prompt(text)}
        ],
        temperature=0.2
    )
    return chat_completion.choices[0].message.content.strip()

# Gradio App

def classify_email(email_text):
    logistic_result = predict_logistic(email_text)
    nb_result = predict_nb(email_text)
    llm_result = predict_llm(email_text)
    return logistic_result, nb_result, llm_result

iface = gr.Interface(
    fn=classify_email,
    inputs=gr.Textbox(lines=10, label="Enter Email Text"),
    outputs=[
        gr.Textbox(label="Logistic Regression Prediction"),
        gr.Textbox(label="Naive Bayes Prediction"),
        gr.Textbox(label="LLM (Groq) Prediction")
    ],
    title="Spam/Ham Email Classifier",
    description="Compare Machine Learning and LLM Predictions on Email Text"
)

if __name__ == "__main__":
    iface.launch()
