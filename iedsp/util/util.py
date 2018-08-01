import base64
import colorsys
import json
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
    with open(filepath, 'wb') as fout:
        pickle.dump(obj, fout)


def load_from_pickle(filepath):
    with open(filepath, 'rb') as fin:
        obj = pickle.load(fin)
    return obj


def save_to_jsonlines(list_of_json, filepath):
    with open(filepath, 'w') as fout:
        for json_obj in list_of_json:
            json_line = json.dumps(json_obj)
            fout.write(json_line + '\n')


def load_from_jsonlines(filepath):
    list_of_json = []
    with open(filepath, 'r') as fin:
        for line in fin.readlines():
            json_obj = json.loads(line)
            list_of_json.append(json_obj)
    return list_of_json


def find_slot_with_key(key, slots):
    # First check is edit or control
    for idx, slot_dict in enumerate(slots):
        if slot_dict['slot'] == key:
            return idx, slot_dict
    return -1, None