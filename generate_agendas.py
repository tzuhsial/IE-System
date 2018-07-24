import argparse
import os
import random
import sys
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, root_dir)

import numpy as np
from pycocotools import mask as maskUtils
from tqdm import tqdm

from iedsp.core import Agent, UserAct, Hermes
from iedsp.ontology import ImageEditOntology
from iedsp.helpers import ActionHelper
from iedsp import util


def annToMask(img, ann):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    Needs to install pycocotools (https://github.com/cocodataset/cocoapi)
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


class ImageEditAgendaGenerator(object):
    """
    Generator that randomly generates agenda
    An agenda is defined as N goals that a user wants to complete in a dialogue session
    ```
    Goal Format
    {
        'dialogue_act': {'slot': 'dialogue_act': 'value': d1 }, 
        'slots': [
            { 'slot': s1, 'value': v1 },
            { 'slot': s2, 'value': v2 }
        ]
    }
    ```
    """

    # Set configurations here for now... Kinda stupid
    ADJUST_PROB = 0.8
    UNDO_PROB = 0.1
    REDO_PROB = 0.1

    # OBJECT
    IMAGE_PROB = 0.2  # Probability of adjusting the whole image

    def __init__(self, dir, num_iers, seed):
        self.dir = dir
        self.num_iers = num_iers
        self.seed = seed

        self._read_from_dir()

        self.ont = ImageEditOntology()

    def _read_from_dir(self):
        """
        Reads imgs & annotations, categories from sampledDir
        """
        # Read original data
        self.imgs = util.load_from_jsonlines(
            os.path.join(self.dir, 'img.jsonl'))
        self.annotations = util.load_from_jsonlines(
            os.path.join(self.dir, 'annotation.jsonl'))
        self.categories = util.load_from_jsonlines(
            os.path.join(self.dir, 'category.jsonl'))

        # Map category id to name
        self.category2name_dict = {}
        for cat in self.categories:
            cat_id = cat['id']
            cat_name = cat['name']
            self.category2name_dict[cat_id] = cat_name

        # Check whether there is mismatch between imgs and annotations
        for img, anns in zip(self.imgs, self.annotations):
            assert all(img['id'] == ann['image_id'] for ann in anns)

    def generate(self):
        """
        Generate random agendas
        """
        # Load seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Return object
        agendas = []

        print("Generating agendas...")
        for img, anns in tqdm(zip(self.imgs, self.annotations)):
            agenda = []

            # IER: open
            image_path = os.path.join(self.dir, 'image', img['file_name'])
            open_slots = [Hermes.build_slot_dict(
                self.ont.Slots.IMAGE_PATH, image_path)]
            open_goal = Hermes.build_act(self.ont.Domains.OPEN, open_slots)
            agenda.append(open_goal)

            # IER: adjust, undo, redo
            domains = [self.ont.Domains.ADJUST,
                       self.ont.Domains.REDO, self.ont.Domains.UNDO]
            domain_probs = [self.ADJUST_PROB, self.REDO_PROB, self.UNDO_PROB]

            for n_ier in range(self.num_iers-2):  # Exclude open & close

                domain_name = np.random.choice(domains, p=domain_probs)
                slot_names = self.ont.getDomainWithName(
                    domain_name).getSlotNames()
                slots = []
                for slot_name in slot_names:
                    if slot_name == "object":
                        """
                        value : {
                            'name': n1,
                            'mask_str': mask_str,
                        }
                        """
                        object_dict = {}
                        object_dict['name'] = 'image'

                        # Add whole image as another option
                        objects = [{}] + anns
                        object_probs = [self.IMAGE_PROB] + \
                            [(1-self.IMAGE_PROB) / len(anns) for _ in anns]
                        object_probs = np.array(object_probs)
                        object_probs /= object_probs.sum()

                        ann = np.random.choice(objects, p=object_probs)
                        if len(ann) > 0:
                            object_name = self.category2name_dict[ann['category_id']]
                            object_mask = annToMask(img, ann)
                            object_mask_str = util.img_to_b64(object_mask)

                            object_dict['name'] = object_name
                            object_dict['mask_str'] = object_mask_str

                        value = object_dict

                    else:
                        """
                        value: v1
                        """
                        possible_values = self.ont.getSlotWithName(
                            slot_name).getPossibleValues()
                        value = random.choice(possible_values)

                    slots.append(Hermes.build_slot_dict(slot_name, value))

                ier_goal = Hermes.build_act(domain_name, slots)
                agenda.append(ier_goal)
            # IER: close
            close_goal = Hermes.build_act(self.ont.Domains.CLOSE)
            agenda.append(close_goal)
            agendas.append(agenda)
        print("Done.")
        return agendas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iers', type=int, default=5,
                        help="Number of image edit requests including open & close")
    parser.add_argument('--dir', type=str, default='./sampled')
    parser.add_argument('--seed', type=int, default=521)
    parser.add_argument('--save', type=str, default='./sampled/agenda.pickle')
    args = parser.parse_args()

    generator = ImageEditAgendaGenerator(args.dir, args.num_iers, args.seed)
    agendas = generator.generate()

    util.save_to_pickle(agendas, args.save)
