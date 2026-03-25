import numpy as np

def get_assymetry(mask):
    """
    Get the assymetry of the image
    :param imgs: list of images
    :return: list of assymetry values
    """
    h, w = mask.shape
    
    left = mask[:, :w//2]
    right = np.fliplr(mask[:, w//2:])
    
    # resize if uneven
    min_w = min(left.shape[1], right.shape[1])
    left = left[:, :min_w]
    right = right[:, :min_w]
    
    diff = np.abs(left - right)
    
    return diff.sum() / mask.sum()


