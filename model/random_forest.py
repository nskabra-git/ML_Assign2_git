from sklearn.ensemble import RandomForestClassifier

def defaults():
    return {"n_estimators": 200, "max_depth": None, "random_state": 42}

def build_model(params=None):
    p = defaults(); p.update(params or {})
    return RandomForestClassifier(**p)
