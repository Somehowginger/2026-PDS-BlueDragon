from skimage import morphology
import numpy as np

def get_compactness(mask):
    # mask = color.rgb2gray(mask)
    mask = (mask > 127)
    area = np.sum(mask)

    struct_el = morphology.disk(3)
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    perimeter = np.sum(mask ^ mask_eroded)

    return round(float(perimeter**2 / (4 * np.pi * area)),3)