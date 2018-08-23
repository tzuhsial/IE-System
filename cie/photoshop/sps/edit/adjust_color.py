import cv2
import numpy as np

from ..utils import random_colors


def get_color_code(color_name):
    color_code_dict = {
        "red": [255, 0, 0],
        "green": [0, 255, 0],
        "blue": [0, 0, 255]
    }
    return color_code_dict.get(color_name, None)


def adjust_color(image, color_name, alpha=0.5):
    """
    Applies color onto the image given the color code
    Args:
        color (ColorCode)
        alpha (float): parameter between image and color
    """
    color = get_color_code(color_name)
    if color is None:
        return None

    colored_image = image.astype(np.uint32).copy()

    for c in range(3):
        colored_image[:, :, c] = image[:, :, c] * \
            (1 - alpha) + alpha * color[c]

    colored_image = colored_image.astype(np.uint8)

    return colored_image
