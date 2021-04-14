### IMPORTS ###
from pycocotools.coco import COCO
import os
import shutil
from argparse import ArgumentParser

### USER ARGUMENTS ###
parser = ArgumentParser(description="Takes COCO image directory, annotations and category as input. Outputs category image to target directory.")
parser.add_argument("-i", action="store", dest="image_dir", type=str, help="path to image directory", required=True)
parser.add_argument("-a", action="store", dest="anno_file", type=str, help="path to annotations for image directory", required=True)
parser.add_argument("-t", action="store", dest="target_dir", type=str, help="target directory", required=True)
parser.add_argument("-c", action="store", dest="category", type=str, help="category", required=True)

args = parser.parse_args()
COCO_PATH = args.image_dir
COCO_ANNOTATIONS_PATH = args.anno_file
COCO_TARGET = args.target_dir
category = args.category

coco = COCO(COCO_ANNOTATIONS_PATH)

# Load categories with the specified ids, in this case all
cats = coco.loadCats(coco.getCatIds())

#get and copy category images
categoryIds = coco.getCatIds(catNms=[category])
imgIds = coco.getImgIds(catIds=categoryIds)
images = coco.loadImgs(imgIds)

for im in images:
    file_name = im["file_name"] 
    input_path = os.path.join(COCO_PATH, file_name)
    target_path = os.path.join(COCO_TARGET, file_name)
#    input_path = COCO_PATH / file_name
#    target_path = COCO_TARGET / file_name
    shutil.copy(input_path, target_path)