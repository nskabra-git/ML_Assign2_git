from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

def load_dataset(as_frame=True):
    data = load_breast_cancer(as_frame=as_frame)
    X = data.data
    y = pd.Series(data.target, name="target")
    feature_names = list(data.feature_names)
    target_names = list(data.target_names)
    return X, y, feature_names, target_names

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def make_scaled(estimator):
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])
