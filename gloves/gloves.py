import os
import numpy as np
import json
import itertools
import detectron2
import glob
# import some common libraries
import numpy as np
import cv2
import random
import pandas as pd
# import some common detectron2 utilities
from detectron2.utils.logger import setup_logger
from detectron2.structures import BoxMode
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode

setup_logger()


# write a function that loads the dataset into detectron2's standard format
def get_balloon_dicts(dataset_dir):
    #   Labels TXT Files (not using)
    '''
    for img_dir in os.listdir(dataset_dir):
        for image_path in glob.glob('{}/*.jpg'.format(img_dir)):
            image = image_path.split('/')[-1]
            image_name = image[:-4]  # without .jpg

            with open(os.path.join('{}/Label/{}'.format(img_dir, image_name)) + '.txt') as f:
                text = f.readlines()
                print(len(text))
                for line in text:
                    split_line = line.split()
                    print(split_line[0])  # Dataset name
                    print(split_line[1])  # coord 1
                    print(split_line[2])  # coord 2
                    print(split_line[3])  # coord 3
                    print(split_line[4])  # coord 4
    '''

    dataset_dicts = []

    # Iteration for dataset classes folders
    for img_dir in os.listdir(dataset_dir):
        class_name: str = ''  # class name like /m/012074

        # Class name
        class_descriptions = 'Dataset/class-descriptions-boxable.csv'
        class_descriptions_boxable = pd.read_csv(class_descriptions)
        for index, row in class_descriptions_boxable.iterrows():
            if row[1] == img_dir:
                class_name = row[0]

        # Image names
        image_names = []
        for image_path in glob.glob('{}/*.jpg'.format(dataset_dir + '/' + img_dir)):
            image = image_path.split('/')[-1]
            image_name = image[:-4]  # without .jpg
            image_names.append(image_name)

        # Annotation CSV Files
        sorted = {}
        dataset_type = dataset_dir.split('/')[1]  # train | validation | test
        annotations_bbox = pd.read_csv('Dataset/' + dataset_type + '-annotations-bbox.csv')  # annotations-bbox.csv
        # getting images that only existing in image_names
        for index, row in annotations_bbox[(annotations_bbox.LabelName == class_name)].iterrows():
            if row['ImageID'] in image_names:
                filename = os.path.join(dataset_dir + '/' + img_dir, row['ImageID'] + '.jpg')
                height, width = cv2.imread(filename).shape[:2]

                x_min = int((width / 100) * int(row['XMin'] * 100) / 1)
                y_min = int((height / 100) * int(row['YMin'] * 100) / 1)
                x_max = int((width / 100) * int(row['XMax'] * 100) / 1)
                y_max = int((height / 100) * int(row['YMax'] * 100) / 1)

                anno = {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [],
                    "category_id": 0,
                    "iscrowd": 0,
                }

                if sorted.get(row['ImageID']):
                    sorted[row['ImageID']]["annotations"].append(anno)
                else:
                    record = {}
                    record["file_name"] = filename
                    record["image_id"] = row['ImageID']
                    record["height"] = height
                    record["width"] = width

                    sorted[row['ImageID']] = record
                    record["annotations"] = [anno]

        for key in sorted:
            dataset_dicts.append(sorted[key])

    return dataset_dicts

'''
# Visualize results
dataset_dicts = get_balloon_dicts("Dataset/train")

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img, metadata=metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
        print(d)
        print(len(d['annotations']))
    cv2.imshow('', vis.get_image()[:, :, ::-1])
    cv2.waitKey()
'''

# Train
cfg = get_cfg()
cfg.merge_from_file("../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("gloves_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (glove)

for d in ["train", "validation"]:
    DatasetCatalog.register("gloves_" + d, lambda d=d: get_balloon_dicts("Dataset/" + d))
    MetadataCatalog.get("gloves_" + d).set(thing_classes=['glove'])

metadata = MetadataCatalog.get("gloves_train")

try:
    model_final_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    open(model_final_path)
    cfg.MODEL.WEIGHTS = model_final_path
    print('model_final exist')
except IOError:
    print('model_final does not exist')
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


# Visualize
dataset_dicts = get_balloon_dicts("Dataset/train")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    # cv2.imshow('', vis.get_image()[:, :, ::-1])
    # cv2.waitKey()

# Predict
cfg.DATASETS.TEST = ("gloves_validation", )
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

predictor = DefaultPredictor(cfg)

dataset_dicts = get_balloon_dicts("Dataset/validation")
for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )

    print(outputs["instances"].to('cpu').pred_boxes)
    # print(outputs["instances"].to("cpu").pred_classes)

    # vis = v.draw_dataset_dict(d)
    # cv2.imshow('', vis.get_image()[:, :, ::-1])
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('', v.get_image()[:, :, ::-1])
    cv2.waitKey()