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


def mask2Dto3D(mask):
    """
    Convert 2D array to 3D by repeat and setting to 255
    """


def main(args):

    imgs = util.load_from_jsonlines(os.path.join(args.dir, 'img.jsonl'))
    annotations = util.load_from_jsonlines(
        os.path.join(args.dir, 'annotation.jsonl'))
    categories = util.load_from_jsonlines(
        os.path.join(args.dir, 'category.jsonl'))
    category2name_dict = {0: 'image'}
    for cat in categories:
        cat_id = cat['id']
        cat_name = cat['name']
        category2name_dict[cat_id] = cat_name

    visionengine = {}

    for img, anns in tqdm(zip(imgs, annotations)):
        image_path = os.path.join(args.dir, 'image', img['file_name'])
        image = util.imread(image_path)
        b64_img_str = util.img_to_b64(image)

        visionengine[b64_img_str] = {}

        for name in category2name_dict.values():
            visionengine[b64_img_str][name] = list()

        for ann in anns:
            object_name = category2name_dict[ann['category_id']]
            one_dim_object_mask = annToMask(img, ann)

            object_mask = np.repeat(one_dim_object_mask[..., np.newaxis], 3, axis=2)\
                .astype(np.uint8)
            indices = object_mask == 1
            object_mask[indices] = 255

            assert ((object_mask == 0) | (object_mask == 255)).all()

            object_mask_str = util.img_to_b64(object_mask)
            visionengine[b64_img_str][object_name].append(object_mask_str)

    util.save_to_pickle(visionengine, args.save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='./sampled')
    parser.add_argument('--save', type=str,
                        default='./sampled/visionengine.annotation.pickle')
    args = parser.parse_args()

    main(args)
