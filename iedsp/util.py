import base64
import colorsys
import pickle

import cv2
import numpy as np


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
    """Converts base64 string back to numpy array
    """
    buf = base64.b64decode(b64_img_str)
    nparr = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def save_to_pickle(obj, filepath):
    """Save to pickle
    """
    with open(filepath, 'wb') as fout:
        pickle.dump(obj, fout)


def load_from_pickle(filepath):
    """Load from pickle
    """
    with open(filepath, 'rb') as fin:
        obj = pickle.load(fin)
    return obj


def find_slot_with_key(key, slots):
    # First check is edit or control
    for idx, slot_dict in enumerate(slots):
        if slot_dict['slot'] == key:
            return idx, slot_dict
    return -1, None


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color=None, alpha=0.5):
    """Apply the given mask to the image.
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
