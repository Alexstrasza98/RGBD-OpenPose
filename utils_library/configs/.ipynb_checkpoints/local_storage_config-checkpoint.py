import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from dataset_config import DATASET, INPUT_TYPE

STORAGE = 'local'

# Definitions for COCO 2017 dataset
if DATASET == 'COCO':
    DATASET_PATH = os.path.dirname(__file__) + "/../../dataset"
    TRAIN_IMAGES_PATH = DATASET_PATH + "/images/train2017"
    VALID_IMAGES_PATH = DATASET_PATH + "/images/val2017"
    TRAIN_ANNS = DATASET_PATH + "/annotations/person_keypoints_train2017.json"
    VALID_ANNS = DATASET_PATH + "/annotations/person_keypoints_val2017.json"

# Definitions for self made dataset
elif DATASET == 'self':
    DATASET_PATH = os.path.dirname(__file__) + "/../../self_dataset"
    
    if INPUT_TYPE == 'RGB':
        DATASET_PATH += "/RGB_Dataset"
        TRAIN_IMAGES_PATH = DATASET_PATH + "/RGB_imgs"
        TRAIN_ANNS = DATASET_PATH + "/RGB_train_annotations.json"
        VALID_IMAGES_PATH = DATASET_PATH + "/RGB_imgs"
        VALID_ANNS = DATASET_PATH + "/RGB_val_annotations.json"
    elif INPUT_TYPE == 'RGBD':
        DATASET_PATH += "/RGBD_Dataset"
        TRAIN_IMAGES_PATH = DATASET_PATH + "/RGBD_imgs"
        TRAIN_ANNS = DATASET_PATH + "/RGBD_train_annotations.json"
        VALID_IMAGES_PATH = DATASET_PATH + "/RGBD_imgs"
        VALID_ANNS = DATASET_PATH + "/RGBD_val_annotations.json"
    else: raise ValueError("Unrecognizable INPUT_TYPE: " + INPUT_TYPE)

else: raise ValueError("Unrecognizable DATASET: " + DATASET)
    
    

# will be used as output TFrecords
if DATASET == 'COCO':
    ROOT_TFRECORDS_PATH = os.path.dirname(__file__) + "/../../dataset/TFrecords"
    TRAIN_TFRECORDS = ROOT_TFRECORDS_PATH + "/training"
    VALID_TFRECORDS = ROOT_TFRECORDS_PATH + "/validation"

elif DATASET == 'self':
    ROOT_TFRECORDS_PATH = os.path.dirname(__file__) + "/../../self_dataset"
    
    if INPUT_TYPE == 'RGB':
        ROOT_TFRECORDS_PATH += "/RGB_Dataset/TFrecords"
    elif INPUT_TYPE == 'RGBD':
        ROOT_TFRECORDS_PATH += "/RGBD_Dataset/TFrecords"
    else: raise ValueError("Unrecognizable INPUT_TYPE: " + INPUT_TYPE)
    
    TRAIN_TFRECORDS = ROOT_TFRECORDS_PATH + "/training"
    VALID_TFRECORDS = ROOT_TFRECORDS_PATH + "/validation"
    
else: raise ValueError("Unrecognizable DATASET: " + DATASET)

# wil be used to save training outputs
RESULTS_ROOT = os.path.dirname(__file__) + "/../../tmp"
TENSORBOARD_PATH = RESULTS_ROOT + "/tensorboard"
CHECKPOINTS_PATH = RESULTS_ROOT + "/checkpoints"
MODELS_PATH = RESULTS_ROOT + "/models"
