# This repository is created for ML classification problem.

# ML Assignment 2 – Classification Models & Streamlit

## Problem Statement
Build and deploy multiple ML classification models using a single dataset and demonstrate them via a Streamlit web application. Deploy on Streamlit Community Cloud and share the live URL.

## Dataset Description
**Breast Cancer Wisconsin (Diagnostic)** dataset (569 samples, 30 numeric features, binary target).
Loaded from `sklearn.datasets.load_breast_cancer`.
Source: Loaded from `sklearn.datasets.load_breast_cancer`

## Models Used and Metrics
Models: Logistic Regression, Decision Tree, KNN, Naive Bayes (Gaussian / Multinomial), Random Forest, XGBoost.

Metrics displayed in the app: Accuracy, AUC, Precision, Recall, F1, **MCC**, Classification Report, Confusion Matrix, ROC and PR curves (when probabilities available).

|        ML Model Name      | Accuracy |   AUC  | Precision | Recall |   F1   |   MCC  |
|---------------------------|---------:|-------:|----------:|-------:|-------:|-------:|
| Logistic Regression       | 0.9649   | 0.9938 | 0.9649    | 0.9649 | 0.9649 | 0.9297 |
| Decision Tree             | 0.9123   | 0.9217 | 0.9217    | 0.9123 | 0.9123 | 0.8258 |
| KNN                       | 0.9561   | 0.9842 | 0.9561    | 0.9561 | 0.9561 | 0.9105 |
| Naive Bayes (Gaussian)    | 0.9386   | 0.9764 | 0.9386    | 0.9386 | 0.9386 | 0.8777 |
| Naive Bayes (Multinomial) | 0.8947   | 0.9351 | 0.8947    | 0.8947 | 0.8947 | 0.7897 |
| Random Forest             | 0.9737   | 0.9928 | 0.9737    | 0.9737 | 0.9737 | 0.9473 |
| XGBoost                   | 0.9737   | 0.9971 | 0.9737    | 0.9737 | 0.9737 | 0.9473 |


## Observations on Model Performance

|        ML Model Name      |              Observation on Model Performance            |   
|---------------------------|---------------------------------------------------------:|
| Logistic Regression       |                                                          |
| Decision Tree             |                                                          |
| KNN                       |                                                          |
| Naive Bayes (Gaussian)    |                                                          |
| Naive Bayes (Multinomial) |                                                          |
| Random Forest             |                                                          |
| XGBoost                   |                                                          |

## Model Info
The model is trained using scikit-learn’s built-in Breast Cancer Wisconsin (Diagnostic) dataset. The Streamlit CSV upload is intended only for test
samples containing feature columns, due to Streamlit free-tier limitations.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py

## Author
Nitin Shriram Kabra

## License
This project is done as part of partial fullfillment of accademic accomplishment.

## Acknowledgements
UCI Machine Learning Repository for the Breast Cancer dataset.

