import base64
import copy

import cv2
import numpy as np


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


def build_slot_dict(slot, value=None, conf=None):
    """
    Slot dict format
    {
        'slot' : s1,
        'value': v1,
        'conf' : c1
    }
    """
    obj = {}
    obj['slot'] = slot
    if value is not None:
        obj['value'] = value
    if conf is not None:
        obj['conf'] = conf
    return obj


def find_slot_with_key(key, slots):
    """
    """
    for idx, slot_dict in enumerate(slots):
        if slot_dict['slot'] == key:
            return slot_dict
    return None


def sort_slots_with_key(key, slots):
    return sorted(slots, key=lambda d: d[key])


def slots_to_args(slots):
    args = {}
    for slot_dict in slots:
        args[slot_dict['slot']] = slot_dict['value']
    return args


def slot_to_observation(slot_dict, turn_id):
    obsrv = copy.deepcopy(slot_dict)
    obsrv.pop('slot')
    obsrv['turn_id'] = turn_id
    return obsrv

