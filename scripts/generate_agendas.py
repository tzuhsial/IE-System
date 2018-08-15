import argparse
import json
import os
import random
import sys
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, root_dir)

import numpy as np
from pycocotools import mask as maskUtils
from tqdm import tqdm

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


def build_goal(intent, slots=None):
    goal = {
        'intent': util.build_slot_dict('intent', intent),
    }
    if slots is not None:
        goal['slots'] = slots
    return goal


def find_ont_with_name(name, ont_slots):
    for ont in ont_slots:
        if ont['name'] == name:
            return ont
    return None


class AdjustAgendaGenerator(object):
    """
    Generator that randomly generates agenda
    An agenda is defined as N goals that a user wants to complete in a dialogue session
    ```
    Goal Format
    {
        'intent': {'slot': 'intent': 'value': d1 },
        'slots': [
            { 'slot': s1, 'value': v1 },
            { 'slot': s2, 'value': v2 }
        ]
    }
    ```
    """

    def __init__(self, ontology_file, num_objects, dir, num_iers, seed):
        with open(ontology_file, 'r') as fin:
            self.ontology_json = json.loads(fin.read())
        self.num_objects = num_objects
        self.dir = dir
        self.num_iers = num_iers
        self.seed = seed

        self._read_from_dir()

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
        # Set 0 to the whole image for convenience
        self.category2name_dict = {0: 'image'}
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
        print("Generating agendas")
        agendas = []
        for img, anns in tqdm(zip(self.imgs, self.annotations)):
            agenda = self.generate_agenda(img, anns)
            agendas.append(agenda)
        return agendas

    def generate_agenda(self, img, anns):
        agenda = []

        # IER: open
        image_path = os.path.join(self.dir, 'image', img['file_name'])
        image_path = os.path.abspath(image_path)
        image_path_slot = util.build_slot_dict('image_path', image_path)
        open_goal = build_goal('open', [image_path_slot])
        agenda.append(open_goal)

        # Get adjust ontology
        ont_slots = self.ontology_json['slots']
        attribute_ont = find_ont_with_name('attribute', ont_slots)
        adjust_value_ont = find_ont_with_name('adjust_value', ont_slots)
        object_ont = find_ont_with_name('object', ont_slots)

        assert attribute_ont is not None
        assert adjust_value_ont is not None
        assert object_ont is not None

        # IER: adjust
        # add the whole image to pool and sample
        # Example:
        # [ dog, cat, mice ]
        # [ 0, 2 ]
        # => [ 0, 0, 1, 2, 2 ]
        # => [ dog, dog, cat, mice, mice ]
        ann_pool = [{'category_id': 0}] + anns
        sampled_anns = np.random.choice(ann_pool, self.num_objects)

        num_duplicates = self.num_iers - self.num_objects
        sampled_duplicates = np.random.choice(
            len(sampled_anns), num_duplicates)
        sampled_duplicates = sorted(sampled_duplicates)
        ann_order = sorted(sampled_duplicates + list(range(len(sampled_anns))))

        final_anns = [sampled_anns[idx] for idx in ann_order]

        assert len(final_anns) == self.num_iers

        for n_ier in range(self.num_iers):

            # Adjust goal has 4 slots
            # 1. attribute
            # Randomly sample an attribute
            attribute_possible_values = attribute_ont['possible_values']

            sampled_attribute = random.choice(attribute_possible_values)

            attribute_slot = util.build_slot_dict(
                'attribute', sampled_attribute)

            # 2. adjust_value
            sampled_adjust_value = random.randint(-50, 50)
            adjust_value_slot = util.build_slot_dict(
                'adjust_value', sampled_adjust_value)

            # 3. object
            object_ann = final_anns[n_ier]
            object_category_id = object_ann['category_id']
            object_name = self.category2name_dict[object_category_id]

            object_slot = util.build_slot_dict('object', object_name)

            # 4. mask_str
            if object_name != "image":
                # From boolean to image
                object_mask = annToMask(img, object_ann)
                indices = object_mask == 1
                object_mask[indices] = 255
                mask_str = util.img_to_b64(object_mask)
                mask_str_slot = util.build_slot_dict(
                    'object_mask_str', mask_str)

                slots = [attribute_slot, adjust_value_slot,
                         object_slot, mask_str_slot]
            else:
                # Without the mask_str_slot
                slots = [attribute_slot, adjust_value_slot, object_slot]

            # Build goal and push to agenda
            adjust_goal = build_goal('adjust', slots)
            agenda.append(adjust_goal)

        # IER: close
        close_goal = build_goal("close")
        agenda.append(close_goal)

        return agenda


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ontology_file', type=str,
                        default='imageedit.ontology.json')
    parser.add_argument('--num_objects', type=int, default=3)
    parser.add_argument('--num_iers', type=int, default=5)
    parser.add_argument('--dir', type=str, default='./sampled')
    parser.add_argument('--seed', type=int, default=521)
    parser.add_argument('--save', type=str, default='./sampled/agenda.pickle')
    args = parser.parse_args()

    assert args.num_iers >= args.num_objects

    generator = AdjustAgendaGenerator(
        args.ontology_file, args.num_objects, args.dir, args.num_iers, args.seed)
    agendas = generator.generate()

    print("Generated {} agendas".format(len(agendas)))
    util.save_to_pickle(agendas, args.save)
