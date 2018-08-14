import base64
import colorsys
import random
import sys

import cv2
if not 'matplotlib' in sys.modules:
    import matplotlib
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import find_contours


def imread(image_path):
    """Provides a wrapper over cv2.imread that converts to RGB space
    """
    bgr_img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img


def img_to_b64(img):
    """Coverts numpy array to base64 image string
    """
    _, nparr = cv2.imencode('.jpg', img)
    b64_img_str = base64.b64encode(nparr).decode()
    return b64_img_str


def b64_to_img(b64_img_str):
    """Converts base64 string back to numpy array using shape
    """
    buf = base64.b64decode(b64_img_str)
    nparr = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


##########################
#   Plotting Functions   #
##########################
"""
Mask plotting functions mostly borrowed from 
https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
"""


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    for idx, tup in enumerate(colors):
        colors[idx] = list(map(lambda c : c * 255, tup))
    return colors


def apply_mask(image, mask, color=None, alpha=0.5):
    """
    Apply the given mask to the image.
    """
    if color is None:
        color = random_colors(1)[0]  # We have only 1 mask at this moment

    # Convert
    masked_image = image.astype(np.uint32).copy()
    boolean_mask = mask.astype('bool').copy()

    for c in range(3):
        condition = boolean_mask[:, :, c] == 1
        masked_channel = image[:, :, c] * (1 - alpha) + alpha * color[c] * 255
        masked_image[:, :, c] = np.where(
            condition, masked_channel, masked_image[:, :, c])

    masked_image = masked_image.astype(np.uint8)

    return masked_image


def plot_diff(img1, img2, cmap=None, figname='diff.png'):
    if img1.ndim == 2:
        cmap = 'gray'
    plt.figure()
    plt.subplot(121)
    plt.imshow(img1, cmap=cmap)
    plt.subplot(122)
    plt.imshow(img2)
    if figname is not None:
        plt.savefig(figname)

def plot_figure(img, figname="test.png"):
    plt.figure()
    plt.imshow(img)
    plt.savefig(figname)


if __name__ == "__main__":
    image_path = "../images/1.jpg"

    imread(image_path)
