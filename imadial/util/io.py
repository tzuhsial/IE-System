import json
import pickle

import cv2


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


def save_to_json(obj, filepath):
    with open(filepath, 'w') as fout:
        json.dump(obj, fout)


def load_from_json(filepath):
    with open(filepath, 'r') as fin:
        obj = json.loads(fin.read())
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


def imread(image_path):
    """Provides a wrapper over cv2.imread that converts to RGB space
    """
    bgr_img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img


def imwrite(image, image_path):
    """converts to bgr format for cv2.imwrite
    """
    bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, bgr_img)
