import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load actual and predicted results
df = pd.read_csv("../results/llm_predictions.csv")

# Normalize labels
df['predicted'] = df['predicted'].str.strip().str.lower()
df['actual'] = df['actual'].str.strip().str.lower()

# Map to binary labels
label_map = {'ham': 0, 'spam': 1}
y_true = df['actual'].map(label_map)
y_pred = df['predicted'].map(label_map)

# Evaluate
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred)

print(f"LLM Accuracy: {accuracy:.2f}")
print("LLM Classification Report:\n", report)

# Save metrics
os.makedirs("../results", exist_ok=True)
with open("../results/llm_metrics.txt", "w") as f:
    f.write(f"LLM Accuracy: {accuracy:.2f}\n")
    f.write("LLM Classification Report:\n")
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title("LLM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("../results/llm_confusion_matrix.png")
plt.close()
