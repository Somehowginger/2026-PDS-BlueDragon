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

    return round(sum(scores) / len(scores),3)