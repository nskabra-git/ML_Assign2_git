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

> Paste the comparison table you export from the app or notebook here.

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

