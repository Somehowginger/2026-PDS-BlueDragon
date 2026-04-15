import cv2
import numpy as np


def convert_to_HSV(img):
    """Convert a BGR image to HSV."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def apply_mask(image, mask):
    """Apply a binary mask to a BGR image and return the masked image."""
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(image, image, mask=mask_bin)


def get_hsv_mean(image, mask=None):
    """Return mean H, S, V values for the masked image.

    If mask is provided, only masked pixels contribute to the mean.
    """
    hsv = convert_to_HSV(image)

    if mask is None:
        h_mean = float(np.mean(hsv[:, :, 0]))
        s_mean = float(np.mean(hsv[:, :, 1]))
        v_mean = float(np.mean(hsv[:, :, 2]))
        return h_mean, s_mean, v_mean

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask_bool = mask_bin.astype(bool)

    h_mean = float(np.mean(hsv[:, :, 0][mask_bool]))
    s_mean = float(np.mean(hsv[:, :, 1][mask_bool]))
    v_mean = float(np.mean(hsv[:, :, 2][mask_bool]))
    return h_mean, s_mean, v_mean
