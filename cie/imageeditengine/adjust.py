"""
    5 attributes
    - Brightness
    - Contrast
    - Hue
    - Saturation
    - Lightness
"""
import cv2
import numpy as np


def adjust(img, attribute, value):
    assert abs(value) <= 100, "Adjustment value should be less than 100!"
    if attribute == "brightness":
        img = adjust_brightness_contrast(img, brightness=value)
    elif attribute == "contrast":
        img = adjust_brightness_contrast(img, contrast=value)
    elif attribute == "hue":
        img = adjust_hue(img, value)
    elif attribute == "saturation":
        img = adjust_saturation(img, value)
    elif attribute == "lightness":
        img = adjust_lightness(img, value)
    else:
        raise AttributeError("hue | saturation | lightness")
    return img


def adjust_brightness_contrast(img, brightness=0, contrast=0):
    """Adjust brightness or contrast
    """
    img = np.int16(img)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    return img


def adjust_hsv_decorator(method):
    def func_wrapper(img, value):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        h, s, v = method(h, s, v, value)
        final_hsv = cv2.merge([h, s, v])
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        return img
    return func_wrapper


@adjust_hsv_decorator
def adjust_hue(h, s, v, value):
    h = cv2.add(h, value)
    return h, s, v


@adjust_hsv_decorator
def adjust_saturation(h, s, v, value):
    s = cv2.add(s, value)
    return h, s, v


@adjust_hsv_decorator
def adjust_lightness(h, s, v, value):
    v = cv2.add(v, value)
    return h, s, v
