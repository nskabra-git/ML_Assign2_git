from sklearn.naive_bayes import GaussianNB, MultinomialNB

def defaults(kind="gaussian"):
    if kind == "multinomial":
        return {"kind": "multinomial", "alpha": 1.0}
    return {"kind": "gaussian"}

def build_model(params=None):
    kind = (params or {}).get("kind", "gaussian")
    p = defaults(kind); 
    if params: p.update(params)
    if p["kind"] == "multinomial":
        return MultinomialNB(alpha=p["alpha"])
    return GaussianNB()
