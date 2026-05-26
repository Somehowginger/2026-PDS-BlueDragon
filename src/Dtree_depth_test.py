from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "results" / "models"

sys.path.append(str(MODEL_DIR))

from decision_tree_classifier import DecisionTree_SkinLeasion_Classifier

features_path = BASE_DIR / "data" / "features.csv"

depths = np.arange(1, 21)
depths = np.round(depths).astype(int)

auc_scores = []

for depth in depths:
    clf = DecisionTree_SkinLeasion_Classifier(max_depth=depth)

    clf.load_and_prepare(features_path)
    clf.train()

    proba = clf.model.predict_proba(clf.X_test)

    classes = list(clf.model.classes_)
    cancerous_index = classes.index("Cancerous")

    prob_cancerous = proba[:, cancerous_index]

    y_true_binary = clf.y_test == "Cancerous"

    auc = roc_auc_score(y_true_binary, prob_cancerous)

    auc_scores.append(auc)

    print(f"max_depth={depth}, AUC={auc:.4f}")


best_index = auc_scores.index(max(auc_scores))
best_depth = list(depths)[best_index]
best_auc = auc_scores[best_index]

print("\nBest max_depth:")
print(f"max_depth={best_depth}, AUC={best_auc:.4f}")


plt.figure(figsize=(8, 5))
plt.plot(list(depths), auc_scores, marker="o")

plt.xlabel("Max Depth")
plt.ylabel("AUC Score")
plt.title("Decision Tree AUC for Different Max Depths")
plt.xticks(list(depths))
plt.ylim(0, 1)
plt.grid(True)

output_path = BASE_DIR / "results" / "figures" / "DTree_AUC_Depths.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nPlot saved to: {output_path}")