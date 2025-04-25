import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# Load predictions
df = pd.read_csv("../results/llm_predictions.csv")

# Normalize label text
df['actual'] = df['actual'].str.strip().str.lower()
df['predicted'] = df['predicted'].str.strip().str.lower()

# Map to binary format
label_map = {'ham': 0, 'spam': 1}
y_true = df['actual'].map(label_map)
y_pred = df['predicted'].map(label_map)

# === Evaluation Metrics ===
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred)

print(f"LLM Accuracy: {accuracy:.2f}")
print("LLM Classification Report:\n", report)

# Save metrics to text file
os.makedirs("../results", exist_ok=True)
with open("../results/llm_metrics.txt", "w") as f:
    f.write(f"LLM Accuracy: {accuracy:.2f}\n\n")
    f.write("LLM Classification Report:\n")
    f.write(report)

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"])
plt.title("LLM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("../results/llm_confusion_matrix.png")
plt.close()

# === ROC Curve ===
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("LLM ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("../results/llm_roc_curve.png")
plt.close()
