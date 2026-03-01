import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)
from src.visualization.utils import ensure_dir

def compute_and_plot_metrics(y_true,y_pred,class_names,output_dir):

    ensure_dir(output_dir)

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true,y_pred,average=None)

    macro_f1 = np.mean(f1)

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_precision": dict(zip(class_names, precision.tolist())),
        "per_class_recall": dict(zip(class_names, recall.tolist()))
    }

    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Precision plot
    plt.figure(figsize=(10, 6))
    plt.barh(class_names, precision)
    plt.xlabel("Precision")
    plt.title("Per-Class Precision")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/per_class_precision.png")
    plt.close()

    # Recall plot
    plt.figure(figsize=(10, 6))
    plt.barh(class_names, recall)
    plt.xlabel("Recall")
    plt.title("Per-Class Recall")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/per_class_recall.png")
    plt.close()

    return metrics
