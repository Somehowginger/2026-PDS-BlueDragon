import numpy as np
from skimage.transform import rotate

def crop(mask):
    y, x = np.nonzero(mask)
    if len(y) == 0:
        return None
    y_min, y_max = y.min(), y.max()
    x_min, x_max = x.min(), x.max()
    return mask[y_min:y_max+1, x_min:x_max+1]


def get_asymmetry(mask):
    mask = mask > 0  # ensure binary

    scores = []

    for angle in range(0, 180, 30):
        rotated = rotate(mask.astype(float), angle, order=0)
        rotated = rotated > 0.5  # re-binarize

        segment = crop(rotated)
        if segment is None or np.sum(segment) == 0:
            continue

        # Horizontal symmetry
        flip_h = np.flip(segment, axis=1)
        asym_h = np.sum(np.logical_xor(segment, flip_h)) / np.sum(segment)

        # Vertical symmetry
        flip_v = np.flip(segment, axis=0)
        asym_v = np.sum(np.logical_xor(segment, flip_v)) / np.sum(segment)

        scores.append((asym_h + asym_v) / 2)

    if len(scores) == 0:
        return 0

    return round(np.mean(scores), 3)