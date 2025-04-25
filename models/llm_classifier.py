import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Choose model
LLM_MODEL = "llama3-8b-8192"

# Prompt builder — LLM gets only email text
def build_prompt(email_text):
    return f"""
You are a cybersecurity assistant. Please classify the following email as either 'Spam' or 'Ham' (not spam).

Email:
\"\"\"
{email_text}
\"\"\"

Your answer must be just one word: Spam or Ham.
"""

# Load dataset
full_df = pd.read_csv("emails.csv")

# Randomly sample N emails
sample_df = full_df.sample(n=10, random_state=42).reset_index(drop=True)

# Extract text for classification, keep actual labels for comparison
email_texts = sample_df['text']
true_labels = sample_df['label']

# LLM classification function
def classify_with_groq(email_text):
    chat_completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for email spam detection."},
            {"role": "user", "content": build_prompt(email_text)}
        ],
        temperature=0.2
    )
    return chat_completion.choices[0].message.content.strip()

# Classify all emails
results = []
print("Running Zero-Shot Spam Detection...\n")

for text, actual_label in tqdm(zip(email_texts, true_labels), total=len(email_texts)):
    prediction = classify_with_groq(text)
    results.append({
        "actual": actual_label,
        "predicted": prediction
    })

# Save predictions
output_df = pd.DataFrame(results)
output_path = "../results/llm_predictions.csv"
os.makedirs("../results", exist_ok=True)
output_df.to_csv(output_path, index=False)

print(f"\n LLM predictions saved to: {output_path}")
