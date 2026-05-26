import cv2
import numpy as np


def removeHair(img_org, img_gray, kernel_size=5, threshold=10, radius=3):
    """Remove hair from image using blackhat morphology + inpainting."""
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    img_out = cv2.inpaint(img_org, hair_mask, radius, cv2.INPAINT_TELEA)
    return blackhat, hair_mask, img_out

def convert_to_HSV(img):
    """Convert a BGR image to HSV."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def apply_mask(image, mask):
    """
    Apply the INVERSE of the lesion mask to isolate skin pixels.
    Returns the masked image (skin only, lesion blacked out).
    """
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = mask.astype(np.uint8)  # FIX: cv2.threshold doesn't accept bool dtype
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    skin_mask = cv2.bitwise_not(mask_bin)   # invert: skin = white, lesion = black

    masked_image = cv2.bitwise_and(image, image, mask=skin_mask)
    return masked_image, skin_mask

def get_hsv_mean(image, mask=None):
    """
    Get mean HSV values, optionally restricted to mask region.
    Mask should be a binary image where WHITE = pixels to include.
    """
    hsv = convert_to_HSV(image)

    if mask is None:
        h_mean = float(np.mean(hsv[:, :, 0]))
        s_mean = float(np.mean(hsv[:, :, 1]))
        v_mean = float(np.mean(hsv[:, :, 2]))
        return round(h_mean, 3), round(s_mean, 3), round(v_mean, 3)

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask_bool = mask.astype(bool)
    h_mean = float(np.mean(hsv[:, :, 0][mask_bool]))
    s_mean = float(np.mean(hsv[:, :, 1][mask_bool]))
    v_mean = float(np.mean(hsv[:, :, 2][mask_bool]))
    return round(h_mean, 3), round(s_mean, 3), round(v_mean, 3)


def pigmentation(image,mask):
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR if image.shape[2] == 4 else cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, _, img_no_hair = removeHair(image, img_gray)

    _, skin_mask = apply_mask(img_no_hair, mask)

    hsv_mean = get_hsv_mean(img_no_hair, skin_mask)
    return(hsv_mean)
