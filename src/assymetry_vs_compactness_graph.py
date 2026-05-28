import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path
from skimage import io
from skimage.transform import resize, rotate
from skimage import morphology


# --- Asymmetry ---
def crop(mask):
    y, x = np.nonzero(mask)
    if len(y) == 0:
        return None
    y_min, y_max = y.min(), y.max()
    x_min, x_max = x.min(), x.max()
    return mask[y_min:y_max+1, x_min:x_max+1]

def get_asymmetry(mask):
    mask = mask > 0
    scores = []
    for _ in range(6):
        segment = crop(mask)
        if segment is None or np.sum(segment) == 0:
            mask = rotate(mask, 30)
            continue
        scores.append(np.sum(np.logical_xor(segment, np.flip(segment))) / np.sum(segment))
        mask = rotate(mask, 30)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 3)


# --- Compactness ---
def get_compactness(mask):
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    area = np.sum(mask)
    if area == 0:
        return None
    struct_el = morphology.disk(3)
    mask_eroded = morphology.erosion(mask, struct_el)
    perimeter = np.sum(mask ^ mask_eroded)
    return round(float(perimeter**2 / (4 * np.pi * area)), 3)


# --- Load and process ---
import pandas as pd
metadata_file = Path(__file__).resolve().parent.parent / "data" / "metadata.csv"
annotations_file = Path(__file__).resolve().parent.parent / "data" / "annotations_combined.csv"

metadata_df = pd.read_csv(metadata_file)
annotations_df = pd.read_csv(annotations_file)
df = metadata_df.merge(annotations_df, on='img_id', how='inner')

rows = []
for _, row in df.iterrows():
    try:
        img_name = row['img_id']
        masks_path = Path(__file__).resolve().parent.parent / "data" / "masks"
        img_path = Path(__file__).resolve().parent.parent / "data" / "imgs"
        mask_name = img_name[:-4] + "_mask" + img_name[-4:]
        mask_path = masks_path / mask_name
        image_path = img_path / img_name

        if not mask_path.exists() or not image_path.exists():
            continue

        mask = io.imread(mask_path)
        img = io.imread(image_path)
        mask1 = resize(mask, img.shape[:2], order=0, preserve_range=True).astype(bool)

        asymmetry = get_asymmetry(mask1)
        compactness = get_compactness(mask1)

        if asymmetry is not None and compactness is not None:
            rows.append({'asymmetry': asymmetry, 'compactness': compactness})

    except Exception as e:
        print(f"Skipping {row['img_id']}: {e}")

result = pd.DataFrame(rows)

# --- Correlation + Plot ---
r, p = pearsonr(result['asymmetry'], result['compactness'])
print(f"Pearson r = {r:.3f}, p-value = {p:.4f}")

plt.figure(figsize=(7, 5))
plt.scatter(result['asymmetry'], result['compactness'], alpha=0.3, s=10, color='steelblue')

m, b = np.polyfit(result['asymmetry'], result['compactness'], 1)
x_line = np.linspace(result['asymmetry'].min(), result['asymmetry'].max(), 200)
plt.plot(x_line, m * x_line + b, color='red', linewidth=1.5, label=f'r = {r:.3f}')

plt.xlabel('Asymmetry')
plt.ylabel('Compactness')
plt.title('Asymmetry vs Compactness')
plt.legend()
plt.tight_layout()

output_path = Path(__file__).resolve().parent.parent / 'results' / 'figures' / "asymmetry_vs_compactness.png"
plt.savefig(output_path, dpi=150)

plt.ylim(0, 100)
output_path = Path(__file__).resolve().parent.parent / 'results' / 'figures' / "asymmetry_vs_compactness_capped.png"
plt.savefig(output_path, dpi=150)