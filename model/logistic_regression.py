from sklearn.linear_model import LogisticRegression
from .data import make_scaled

def defaults():
    return {"C": 1.0, "solver": "lbfgs", "max_iter": 500}

def build_model(params=None):
    p = defaults(); p.update(params or {})
    base = LogisticRegression(C=p["C"], solver=p["solver"], max_iter=p["max_iter"])
    return make_scaled(base)
