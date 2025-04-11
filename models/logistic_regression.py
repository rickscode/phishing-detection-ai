import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
csv_file = "../data/emails.csv"
df = pd.read_csv(csv_file)

# Set proper column names
df.columns = ['id', 'label', 'text', 'label_num']

# Extract features and labels
X = df['text']
y = df['label_num']  # 0 for ham, 1 for spam

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to numerical form using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Logistic Regression Model Accuracy:", accuracy)
print("\nClassification Report:\n", report)

# Save model and vectorizer
joblib.dump(model, "logistic_regression_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Save metrics to text file
os.makedirs("../results", exist_ok=True)
with open("../results/lr_model_metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write("Classification Report:\n")
    f.write(report)

# === Visualization ===

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot(cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")
plt.savefig("../results/lr_confusion_matrix.png")
plt.close()

# 2. Classification Report Heatmap
report_dict = classification_report(y_test, y_pred, output_dict=True)
sns.heatmap(pd.DataFrame(report_dict).iloc[:-1, :].T, annot=True, cmap="YlGnBu")
plt.title("Logistic Regression - Classification Report Heatmap")
plt.savefig("../results/lr_report_heatmap.png")
plt.close()

# 3. ROC Curve
y_prob = model.predict_proba(X_test_tfidf)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression - ROC Curve")
plt.legend(loc="lower right")
plt.savefig("../results/lr_roc_curve.png")
plt.close()

print("Visualizations saved to ../results/")
