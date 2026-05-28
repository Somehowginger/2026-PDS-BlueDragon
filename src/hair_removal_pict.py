import cv2
import matplotlib as plt

def removeHair(img_org, img_gray, kernel_size=5, threshold=10, radius=3):
    """Remove hair from image using blackhat morphology + inpainting."""
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    img_out = cv2.inpaint(img_org, hair_mask, radius, cv2.INPAINT_TELEA)
    return blackhat, hair_mask, img_out

def picture_gen(image, mask):
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR if image.shape[2] == 4 else cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, _, img_no_hair = removeHair(image, img_gray)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_no_hair )
    plt.title("Hair-Removed Image")
    plt.axis("off")
    plt.show()

imageObj = cv2.imread('../data/imgs/PAT_20_30_44.png')
maskObj = cv2.imread("../data/masks/PAT_20_30_44_mask.png")
picture_gen(imageObj, maskObj)