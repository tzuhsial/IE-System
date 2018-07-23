"""
    Perform filter options
    - black_and_white
    - gaussian
"""
import cv2


def add_filter(img, filter_type, value=10):
    """Apply a filter to the image
    """
    if filter_type == "black_and_white":
        # Convert to black and white scale
        grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        (_, filtered_img) = cv2.threshold(grey_img,
                                          128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    elif filter_type == "gaussian":
        filtered_img = cv2.GaussianBlur(img, (5, 5), 0)
    else:
        raise ValueError("Unknown filter: {}!".format(filter_type))
    """
    elif filter_type == "box":
        filtered_img = cv2.blur(img, (5,5))
    elif filter_type == "median":
        filtered_img = cv2.medianBlur(img,5)
    elif filter_type == "bilateral":
        filtered_img = cv2.bilateralFilter(img,9,75,75)
    """
    return filtered_img
