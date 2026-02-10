import io, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, streamlit as st

from model.data import load_dataset, split_data
from model.evaluate import compute_metrics, curves, confusion, as_dataframe, report_text
from model.plots import plot_confusion_matrix, plot_roc, plot_pr

# Model builders + defaults
from model.logistic_regression import build_model as build_lr, defaults as d_lr
from model.decision_tree import build_model as build_dt, defaults as d_dt
from model.knn import build_model as build_knn, defaults as d_knn
from model.naive_bayes import build_model as build_nb, defaults as d_nb
from model.random_forest import build_model as build_rf, defaults as d_rf
from model.xgboost_model import build_model as build_xgb, defaults as d_xgb

st.set_page_config(page_title="ML Classifiers Demo", layout="wide")
st.title("Classification Models — Breast Cancer (Diagnostic)")

X, y, feature_names, target_names = load_dataset()
feature_names_norm = [
    f.lower().replace(" ", "_") for f in feature_names
]
with st.expander("About the dataset"):
    st.markdown("Using scikit‑learn's **Breast Cancer Wisconsin (Diagnostic)** dataset (569 samples, 30 features).")

# ---------- CSV upload for TEST data (assignment: upload only test) ----------
st.sidebar.header("1) Optional: Upload TEST CSV")
up = st.sidebar.file_uploader("Upload CSV with the same 30 feature columns (no target).", type=["csv"])

st.sidebar.header("2) Train/Test Split (if no CSV)")
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)

st.sidebar.caption(
    "Uploaded test CSV must follow the same feature order as the dataset. "
    "Extra columns are ignored."
)

if up is None:
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=42)
else:
    # Always treat uploaded CSV as HEADERLESS
    up.seek(0)
    df_test = pd.read_csv(up, header=None)

    # Allow extra columns; require at least the required number of features
    if df_test.shape[1] < len(feature_names_norm):
        st.error(
            f"Uploaded CSV must have at least {len(feature_names_norm)} columns, "
            f"but got {df_test.shape[1]}"
        )
        st.stop()

    # Use only the first required feature columns
    df_test = df_test.iloc[:, :len(feature_names_norm)]

    # Assign correct feature names
    df_test.columns = feature_names

    X_test = df_test
    X_train, y_train = X, y
    y_test = None


# ---------- Model selection ----------
st.sidebar.header("3) Choose Model & Hyperparameters")
choices = [
    "Logistic Regression", "Decision Tree", "KNN",
    "Naive Bayes (Gaussian)", "Naive Bayes (Multinomial)",
    "Random Forest", "XGBoost"
]
name = st.sidebar.selectbox("Classifier", choices)

params = {}
if name == "Logistic Regression":
    d = d_lr()
    params["C"] = st.sidebar.slider("C", 0.01, 10.0, float(d["C"]), 0.01)
    params["solver"] = st.sidebar.selectbox("solver", ["lbfgs","liblinear","saga"], index=0)
    params["max_iter"] = st.sidebar.slider("max_iter", 100, 1000, int(d["max_iter"]), 50)
elif name == "Decision Tree":
    d = d_dt()
    params["max_depth"] = st.sidebar.slider("max_depth", 1, 20, 5 if d["max_depth"] is None else int(d["max_depth"]), 1)
    params["min_samples_split"] = st.sidebar.slider("min_samples_split", 2, 20, int(d["min_samples_split"]), 1)
elif name == "KNN":
    d = d_knn()
    params["n_neighbors"] = st.sidebar.slider("n_neighbors", 1, 25, int(d["n_neighbors"]), 1)
    params["weights"] = st.sidebar.selectbox("weights", ["uniform","distance"], index=0)
    params["metric"] = st.sidebar.selectbox("metric", ["minkowski","euclidean","manhattan"], index=0)
elif name == "Naive Bayes (Gaussian)":
    params["kind"] = "gaussian"
elif name == "Naive Bayes (Multinomial)":
    d = d_nb("multinomial")
    params["kind"] = "multinomial"
    params["alpha"] = st.sidebar.slider("alpha", 0.0, 5.0, float(d["alpha"]), 0.1)
elif name == "Random Forest":
    d = d_rf()
    params["n_estimators"] = st.sidebar.slider("n_estimators", 50, 500, int(d["n_estimators"]), 50)
    params["max_depth"] = st.sidebar.slider("max_depth", 1, 30, 10 if d["max_depth"] is None else int(d["max_depth"]), 1)
elif name == "XGBoost":
    d = d_xgb()
    params["n_estimators"] = st.sidebar.slider("n_estimators", 50, 800, int(d["n_estimators"]), 50)
    params["learning_rate"] = st.sidebar.slider("learning_rate", 0.01, 0.5, float(d["learning_rate"]), 0.01)
    params["max_depth"] = st.sidebar.slider("max_depth", 2, 12, int(d["max_depth"]), 1)
    params["subsample"] = st.sidebar.slider("subsample", 0.5, 1.0, float(d["subsample"]), 0.05)
    params["colsample_bytree"] = st.sidebar.slider("colsample_bytree", 0.5, 1.0, float(d["colsample_bytree"]), 0.05)

# run_btn = st.sidebar.button("Run")
run_btn = st.sidebar.button("Run", type="primary")
auto_run = st.sidebar.checkbox(
    "Auto-run with built-in dataset", 
    value=True,
    help="Runs evaluation automatically using the default dataset"
)

def get_builder(nm):
    return {
        "Logistic Regression": build_lr,
        "Decision Tree": build_dt,
        "KNN": build_knn,
        "Naive Bayes (Gaussian)": lambda p: build_nb({"kind":"gaussian"}),
        "Naive Bayes (Multinomial)": lambda p: build_nb({"kind":"multinomial", **p}),
        "Random Forest": build_rf,
        "XGBoost": build_xgb,
    }[nm]

# if run_btn:
if run_btn or (auto_run and up is None):
    if y_test is None:
        st.info(
            "Uploaded CSV has no ground-truth labels. "
           "Showing model predictions only."
        )

        model = get_builder(name)(params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        # Map numeric predictions to class names
        pred_labels = [target_names[p] for p in preds]

        results = X_test.copy()
        results["Prediction"] = pred_labels

        st.subheader("Predictions on Uploaded Test Data")
        st.dataframe(results)

        st.stop()

    model = get_builder(name)(params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

    # Metrics incl. MCC
    m = compute_metrics(y_test, y_pred, y_proba)
    st.subheader("Evaluation Metrics (incl. MCC)")
    st.dataframe(as_dataframe(m).style.format("{:.4f}"))

    # Confusion Matrix
    cm = confusion(y_test, y_pred)
    st.subheader("Confusion Matrix")
    st.pyplot(plot_confusion_matrix(cm, labels=[str(t) for t in target_names]))

    # Classification Report
    st.subheader("Classification Report")
    report = report_text(y_test, y_pred, target_names=[str(t) for t in target_names])
    st.code(report, language="text")
