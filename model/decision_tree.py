from sklearn.tree import DecisionTreeClassifier

def defaults():
    return {"max_depth": 5, "min_samples_split": 2, "random_state": 42}

def build_model(params=None):
    p = defaults(); p.update(params or {})
    return DecisionTreeClassifier(**p)
