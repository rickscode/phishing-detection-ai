import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 1: Load the dataset
csv_file = "../data/emails.csv"
df = pd.read_csv(csv_file)

# Step 2: Extract features and labels
X = df["text"]  # Email content
y = df["label_num"]  # 0 = Ham, 1 = Spam

# Step 3: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train a Naïve Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

# Save model and vectorizer
joblib.dump(model, "naives_bayes_model.pkl")
joblib.dump(vectorizer, "nb_tfidf_vectorizer.pkl")

# Save text metrics
os.makedirs("../results", exist_ok=True)
with open("../results/nb_model_metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write("Classification Report:\n")
    f.write(report)

# === Visualization ===

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot(cmap="Blues")
plt.title("Naïve Bayes - Confusion Matrix")
plt.savefig("../results/nb_confusion_matrix.png")
plt.close()

# 2. Classification Report Heatmap
report_dict = classification_report(y_test, y_pred, output_dict=True)
sns.heatmap(pd.DataFrame(report_dict).iloc[:-1, :].T, annot=True, cmap="YlGnBu")
plt.title("Naïve Bayes - Classification Report Heatmap")
plt.savefig("../results/nb_report_heatmap.png")
plt.close()

# 3. ROC Curve
y_prob = model.predict_proba(X_test_tfidf)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Naïve Bayes - ROC Curve")
plt.legend(loc="lower right")
plt.savefig("../results/nb_roc_curve.png")
plt.close()

print("Visualizations saved to results/")
