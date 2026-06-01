# Breast Cancer Detection - ML Classification

 Model Machine Learning pentru detectarea tumorilor 
maligne/benigne folosind 7 algoritmi de clasificare.

## Dataset
- Sursa: sklearn built-in (Wisconsin Breast Cancer Dataset)
- 569 paciente, 30 caracteristici medicale
- Target: malign (0) / benign (1)

## Rezultate

| Model | Acuratețe |
|-------|-----------|
| Naive Bayes | 97.4% |
| Random Forest | 96.5% |
| Gradient Boosting | 95.6% |
| SVM | 94.7% |

## Ce am aplicat
- Explorare și analiză date
- Feature Importance (Random Forest)
- Feature Scaling (StandardScaler)
- Comparație 7 modele ML

## Tehnologii
Python · scikit-learn · Pandas · NumPy

## Concluzie
Naive Bayes câștigă datorită independenței relative
a măsurătorilor medicale — model simplu, rezultat superior.
