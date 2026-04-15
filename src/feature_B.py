from skimage import morphology
import numpy as np

def get_compactness(mask):
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = (mask > 127)
    area = np.sum(mask)

    if area == 0:
        return None

    struct_el = morphology.disk(3)
    mask_eroded = morphology.erosion(mask, struct_el)
    perimeter = np.sum(mask ^ mask_eroded)

    return round(float(perimeter**2 / (4 * np.pi * area)), 3)