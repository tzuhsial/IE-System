import os
import random
import sys
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, root_dir)

import numpy as np
from pycocotools import mask as maskUtils

from iedsp.ontology import Ontology
from iedsp import util

random.seed(521)
np.random.seed(521)

"""
    Goal (dict)
    keys
    - intent (str)
    - slots (list)
"""


def annToMask(img, ann):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    h, w = img['height'], img['width']
    segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']
    m = maskUtils.decode(rle)
    return m


def generate_random_agenda(img, anns, categories, sampledDir):
    """Generate image edit requests with Ontology
       Agenda is a list of goals

    """
    num_of_iers = 5

    agenda = list()

    # Open an image first
    imgPath = os.path.join(sampledDir, 'image', img['file_name'])

    open_goal = {}
    open_goal['intent'] = 'open'
    open_goal['slots'] = [{'slot': 'image_path', 'value': imgPath}]
    agenda.append(open_goal)

    # 3 image edit request
    category2name_dict = {}
    for cat in categories:
        cat_id = cat['id']
        cat_name = cat['name']
        category2name_dict[cat_id] = cat_name

    for nier in range(num_of_iers):
        ier_goal = {}
        ier_goal['intent'] = np.random.choice(
            ['adjust', 'undo', 'redo'], p=[0.8, 0.1, 0.1])
        intent_slots = Ontology.getArgumentsWithIntent(ier_goal['intent'])

        slots = []
        for slot_name in intent_slots:

            if slot_name == "object":

                object_dict = {}
                object_dict['name'] = 'image'

                # Add whole image as another option
                object_probs = np.array(
                    [0.2] + [(0.8 / len(anns)) for _ in anns])

                object_probs /= object_probs.sum()

                # assert object_probs.sum() == 1., "{} sum: {}".format(
                #    object_probs, object_probs.sum())
                ann = np.random.choice([{}] + anns, p=object_probs)
                if len(ann) > 0:
                    object_name = category2name_dict[ann['category_id']]
                    object_mask = annToMask(img, ann)
                    object_mask_str = util.img_to_b64(object_mask)

                    object_dict['name'] = object_name
                    object_dict['mask_str'] = object_mask_str

                value = object_dict

            else:
                possibleValues = Ontology.getPossibleValuesWithSlot(slot_name)
                value = random.choice(possibleValues)

            slot_dict = {'slot': slot_name, 'value': value}
            slots.append(slot_dict)

        ier_goal['slots'] = slots

        agenda.append(ier_goal)

    # Close image
    close_goal = {}
    close_goal['intent'] = 'close'
    close_goal['slots'] = []
    agenda.append(close_goal)

    return agenda


if __name__ == "__main__":

    sampledDir = './sampled'

    sampledImgs = util.load_from_jsonlines(
        os.path.join(sampledDir, 'img.jsonl'))
    sampledAnnotations = util.load_from_jsonlines(
        os.path.join(sampledDir, 'annotation.jsonl'))
    categories = util.load_from_jsonlines(
        os.path.join(sampledDir, 'category.jsonl'))

    # Check whether sampledImg and sampledAnnotations mismatch
    for img, anns in zip(sampledImgs, sampledAnnotations):
        assert all(img['id'] == ann['image_id'] for ann in anns)

    agendas = []
    for img, anns in zip(sampledImgs, sampledAnnotations):
        agenda = generate_random_agenda(img, anns, categories, sampledDir)
        agendas.append(agenda)

    util.save_to_pickle(agendas, os.path.join(sampledDir, 'agendas.pickle'))
