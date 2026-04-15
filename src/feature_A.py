import numpy as np
from skimage.transform import rotate

def crop(mask):
    # find where mask is non-zero (lesion pixels)
    y, x = np.nonzero(mask)

    # get bounding box
    y_min, y_max = y.min(), y.max()
    x_min, x_max = x.min(), x.max()

    # crop
    return mask[y_min:y_max+1, x_min:x_max+1]

def get_asymmetry(mask):
    scores = []

    for _ in range(6):
        segment = crop(mask)
        (np.sum(segment))
        scores.append(np.sum(np.logical_xor(segment, np.flip(segment))) / (np.sum(segment)))
        mask = rotate(mask, 30)

    return sum(scores) / len(scores)

