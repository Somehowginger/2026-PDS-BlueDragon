import cv2
import numpy as np


def convert_to_HSV(img):
    """Convert a BGR image to HSV."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def get_hsv_mean(image, mask=None):
    hsv = convert_to_HSV(image)

    if mask is None:
        h_mean = float(np.mean(hsv[:, :, 0]))
        s_mean = float(np.mean(hsv[:, :, 1]))
        v_mean = float(np.mean(hsv[:, :, 2]))
        return round(h_mean, 3), round(s_mean, 3), round(v_mean, 3)

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    mask_bool = mask.astype(bool)  # applying mask

    h_mean = float(np.mean(hsv[:, :, 0][mask_bool]))
    s_mean = float(np.mean(hsv[:, :, 1][mask_bool]))
    v_mean = float(np.mean(hsv[:, :, 2][mask_bool]))
    return round(h_mean,3), round(s_mean,3), round(v_mean,3)