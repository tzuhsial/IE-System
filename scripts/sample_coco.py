import json
import os
import random
import shutil
import sys
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, root_dir)

from pycocotools.coco import COCO

from iedsp import util

random.seed(521)  # The date I went on board


if __name__ == "__main__":
    # Define arguments here
    dataDir = os.path.join(root_dir, 'data', 'coco')
    dataType = 'train2017'
    numOfImages = 1000
    saveDir = './sampled'

    # Get paths
    annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)
    imgDir = '%s/images' % (dataDir)

    # Get COCO class object
    coco = COCO(annFile)

    # Get all img ids
    allImgIds = sorted(coco.getImgIds())

    random.shuffle(allImgIds)

    sampledImgs = [coco.loadImgs(imgId)[0]
                   for imgId in allImgIds[:numOfImages]]

    sampledAnnotationsIds = [coco.getAnnIds(
        imgIds=img['id']) for img in sampledImgs]
    sampledAnnotations = [coco.loadAnns(annIds)
                          for annIds in sampledAnnotationsIds]

    # Save to directory
    saveImgDir = os.path.join(saveDir, 'image')
    if not os.path.exists(os.path.join(saveImgDir)):
        os.makedirs(saveImgDir)

    # Save category names
    category_filepath = os.path.join(saveDir, 'category.jsonl')
    cats = coco.loadCats(coco.getCatIds())
    util.save_to_jsonlines(cats, category_filepath)

    # Save sampled img ids
    img_filepath = os.path.join(saveDir, 'img.jsonl')
    util.save_to_jsonlines(sampledImgs, img_filepath)

    # Save Annotations
    annotation_filepath = os.path.join(saveDir, 'annotation.jsonl')
    util.save_to_jsonlines(sampledAnnotations, annotation_filepath)

    # Move to images to directory
    for img in sampledImgs:
        src = os.path.join(imgDir, img['file_name'])
        dst = os.path.join(saveImgDir, img['file_name'])
        shutil.copy(src, dst)
