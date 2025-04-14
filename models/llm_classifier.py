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
LLM_MODEL = "llama3-8b-8192"  # Can change to "llama3-70b-8192" if needed

# Prompt builder
def build_prompt(email_text):
    return f"""
You are a cybersecurity assistant. Please classify the following email as either 'Spam' or 'Ham' (not spam).

Email:
\"\"\"
{email_text}
\"\"\"

Your answer must be just one word: Spam or Ham.
"""

# Load sample emails
df = pd.read_csv("emails.csv")
df = df[['text', 'label']].sample(n=10, random_state=42)

# Classify using Groq LLM
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

# Run classification loop
results = []
print("🧠 Running Zero-Shot LLM Spam Detection...\n")

for _, row in tqdm(df.iterrows(), total=len(df)):
    prediction = classify_with_groq(row["text"])
    results.append({
        "actual": row["label"],
        "predicted": prediction
    })

# Save predictions
output_df = pd.DataFrame(results)
output_path = "../results/llm_predictions.csv"
output_df.to_csv(output_path, index=False)

print(f"\n✅ LLM predictions saved to: {output_path}")
