# Breast Cancer Detection — ML Classification Pipeline

A Machine Learning diagnostic pipeline designed to classify breast tumor biopsies as **Malignant** or **Benign** by benchmarking multiple classification algorithms.

---

## Overview & Dataset

* **Source:** Wisconsin Breast Cancer Diagnostic Dataset (`sklearn.datasets.load_breast_cancer`)
* **Samples:** 569 patient instances
* **Features:** 30 continuous medical attributes extracted from digitized fine needle aspirate (FNA) images (e.g., radius, texture, perimeter, area, smoothness, concavity)
* **Target Classes:** 
  * `0`: Malignant
  * `1`: Benign

---

## Technical Methodology

* **Data Preprocessing & Splitting:** Stratified 80/20 train/test split to preserve natural class balance across sets.
* **Feature Scaling:** `StandardScaler` fitted strictly on training data and transformed across test data to prevent data leakage and optimize margin/distance-based algorithms (SVM, Logistic Regression).
* **Multi-Model Benchmark:** Evaluated and tuned multiple supervised models (Gaussian Naive Bayes, Random Forest, Gradient Boosting, Support Vector Classifier, Logistic Regression).
* **Feature Importance:** Analyzed Gini impurity decreases via Random Forest to identify primary clinical predictors (`worst perimeter`, `worst concave points`, `worst radius`).
* **Medical Metric Priority:** Special emphasis placed on **Recall (Sensitivity)** for malignant cases to minimize critical false negatives in diagnostic screening.

---

## Benchmark Results

| Model | Accuracy |
| :--- | :---: |
| **Gaussian Naive Bayes** | **97.37%** |
| **Random Forest Classifier** | **96.49%** |
| **Gradient Boosting** | **95.61%** |
| **Support Vector Classifier (SVC)** | **94.74%** |

---

## Tech Stack

* **Language:** Python 3
* **Libraries:** scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

---

## Execution Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/dddionut1/breast-cancer-ml.git](https://github.com/dddionut1/breast-cancer-ml.git)
   cd breast-cancer-ml
