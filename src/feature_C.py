import cv2

def convert_to_HSV(imgs):
# Convert to HSV
    hsv_image = cv2.cvtColor(imgs, cv2.COLOR_BGR2LAB)
    return hsv_image