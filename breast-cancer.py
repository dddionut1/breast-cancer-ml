import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB


# datele
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

'''print(df.shape)
print(df.head())
print(df["target"].value_counts())
print(df.isnull().sum().sum())  # trebuie să fie 0'''


# X = toate coloanele EXCEPT target
X = df.drop("target", axis=1)

# y = DOAR coloana target
y = df["target"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training: {len(X_train)} paciente")
print(f"Testing:  {len(X_test)} paciente")

#Random forest
print("Random Forest")
model = RandomForestClassifier(n_estimators=100, random_state=42)#creare

model.fit(X_train, y_train)#antrenare

predictii = model.predict(X_test)#prezicere


print(f"Acuratete: {accuracy_score(y_test, predictii) * 100:.1f}%") #evaluare
print(classification_report(y_test, predictii))

importanta = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
print("\nCe conteaza cel mai mult:")
print(importanta)


modele_noi = {
    "SVM":                 (SVC(random_state=42), X_train, X_test),
    "Gradient Boosting":   (GradientBoostingClassifier(random_state=42), X_train, X_test),
    "Naive Bayes":         (GaussianNB(), X_train, X_test),
}

print("\n=== Modele noi — Breast Cancer ===")
for nume, (model, Xtr, Xte) in modele_noi.items():
    model.fit(Xtr, y_train)
    acc = accuracy_score(y_test, model.predict(Xte))
    print(f"{nume:25} → {acc * 100:.1f}%")


