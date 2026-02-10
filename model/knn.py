from sklearn.neighbors import KNeighborsClassifier
from .data import make_scaled

def defaults():
    return {"n_neighbors": 5, "weights": "uniform", "metric": "minkowski"}

def build_model(params=None):
    p = defaults(); p.update(params or {})
    base = KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights=p["weights"], metric=p["metric"])
    return make_scaled(base)
