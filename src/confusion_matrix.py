import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path


def plot_confusion_matrix_from_csv(csv_path, model_name, output_path):
    # Load predictions
    df = pd.read_csv(csv_path)

    # True labels and predicted labels
    y_true = df["Actual"]
    y_pred = df["Prediction"]

    # Change these if your labels are named differently
    labels = ["Benign", "Cancerous"]

    # These are the labels shown on the plot
    display_labels = ["Benign", "Cancer"]

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=display_labels
    )

    disp.plot(values_format="d", cmap=plt.cm.Blues)

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    # Save plot
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.show()

    print(f"{model_name} confusion matrix saved to: {output_path}")


# Go to repository root
BASE_DIR = Path(__file__).resolve().parents[1]

# Paths to prediction CSV files
knn_predictions = BASE_DIR / "results" / "predictions" / "predictions_knn.csv"
dt_predictions = BASE_DIR / "results" / "predictions" / "predictions_decisiontree.csv"

# Output folder
output_dir = BASE_DIR /"results" /"figures"
output_dir.mkdir(parents=True, exist_ok=True)

#Create confusion matrices
plot_confusion_matrix_from_csv(
    knn_predictions,
    "KNN Classifier",
    output_dir / "knn_confusion_matrix.png"
)

plot_confusion_matrix_from_csv(
    dt_predictions,
    "Decision Tree Classifier",
    output_dir / "decision_tree_confusion_matrix-3F.png"
)
