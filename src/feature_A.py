import numpy as np
from skimage.transform import rotate
from skimage.util import crop

def get_asymmetry(mask):
    scores = []

    for _ in range(6):
        segment = crop(mask)
        (np.sum(segment))
        scores.append(np.sum(np.logical_xor(segment, np.flip(segment))) / (np.sum(segment)))
        mask = rotate(mask, 30)

    return sum(scores) / len(scores)
