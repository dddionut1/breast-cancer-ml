import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

# Features (X) and Target (y)
X = df.drop("target", axis=1)
y = df["target"]

# 2. Train / Test Split (Stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}\n")

# 3. Feature Scaling (Critical for distance/margin-based algorithms like SVM & Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model Training & Comparison
models = {
    "Gaussian Naive Bayes": (GaussianNB(), X_train, X_test),
    "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42), X_train, X_test),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42), X_train, X_test),
    "Support Vector Machine (SVC)": (SVC(random_state=42), X_train_scaled, X_test_scaled),
    "Logistic Regression": (LogisticRegression(random_state=42, max_iter=1000), X_train_scaled, X_test_scaled)
}

print("=== Model Benchmark Results ===")
best_model_name = ""
best_acc = 0.0
best_model_instance = None

for name, (model, xtr, xte) in models.items():
    model.fit(xtr, y_train)
    preds = model.predict(xte)
    acc = accuracy_score(y_test, preds)
    print(f"{name:30} -> Accuracy: {acc * 100:.2f}%")
    
    if acc > best_acc:
        best_acc = acc
        best_model_name = name
        best_model_instance = (model, xte)

# 5. Detailed Evaluation of Top Performer
print(f"\n=== Detailed Evaluation: {best_model_name} ===")
best_preds = best_model_instance[0].predict(best_model_instance[1])
print("Confusion Matrix:")
print(confusion_matrix(y_test, best_preds))
print("\nClassification Report (Recall prioritized for medical diagnosis):")
print(classification_report(y_test, best_preds, target_names=["Malignant (0)", "Benign (1)"]))

# 6. Feature Importance (via Random Forest)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("Top 5 Most Predictive Features (Random Forest):")
print(feature_importances.head(5))
