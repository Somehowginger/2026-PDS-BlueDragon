import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

features_path = BASE_DIR / "data" / "features.csv"
masks_dir = BASE_DIR / "data" / "masks"
output_dir = BASE_DIR / "results" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Column names
# Change these if your CSV uses different names
# -----------------------------
IMAGE_COL = "Image_ID"
ASYM_COL = "A_asymmetry"
LABEL_COL = "Cancer"

BENIGN_LABEL = "Benign"
CANCER_LABEL = "Cancerous"


# -----------------------------
# Helper: get mask path from Image_ID
# -----------------------------
def get_mask_path(image_id):
    image_id = str(image_id)

    # remove file extension if Image_ID contains .jpg/.png
    image_stem = Path(image_id).stem

    # example: PAT_8_15_820 -> PAT_8_15_820_mask.png
    return masks_dir / f"{image_stem}_mask.png"


# -----------------------------
# Load feature file
# -----------------------------
df = pd.read_csv(features_path)

# Pick 3 benign and 3 cancerous examples
benign = df[df[LABEL_COL] == BENIGN_LABEL].sample(3, random_state=42)
cancerous = df[df[LABEL_COL] == CANCER_LABEL].sample(3, random_state=42)

examples = pd.concat([benign, cancerous])


# -----------------------------
# Plot masks
# -----------------------------
fig, axes = plt.subplots(2, 3, figsize=(12, 7))

for ax, (_, row) in zip(axes.ravel(), examples.iterrows()):
    image_id = row[IMAGE_COL]
    label = row[LABEL_COL]
    asymmetry = row[ASYM_COL]

    mask_path = get_mask_path(image_id)

    mask = io.imread(mask_path, as_gray=True)

    ax.imshow(mask, cmap="gray")
    ax.set_title(f"{label}\nAsymmetry: {asymmetry:.3f}")
    ax.axis("off")


plt.suptitle("Comparison of Asymmetry Scores", fontsize=14)
plt.tight_layout()

output_path = output_dir / "asymmetry_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Plot saved to: {output_path}")