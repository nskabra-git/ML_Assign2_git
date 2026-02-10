import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    fig.tight_layout(); return fig

def plot_roc(fpr, tpr):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC"); ax.plot([0,1], [0,1], "--", color="gray")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve"); ax.legend()
    fig.tight_layout(); return fig

def plot_pr(prec, rec):
    fig, ax = plt.subplots()
    ax.plot(rec, prec, label="PR")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision–Recall Curve"); ax.legend()
    fig.tight_layout(); return fig
