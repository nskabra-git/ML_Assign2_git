from xgboost import XGBClassifier

def defaults():
    return {
        "n_estimators": 200, "learning_rate": 0.1, "max_depth": 4,
        "subsample": 0.9, "colsample_bytree": 0.9, "tree_method": "hist",
        "eval_metric": "logloss", "random_state": 42
    }

def build_model(params=None):
    p = defaults(); p.update(params or {})
    return XGBClassifier(**p)
