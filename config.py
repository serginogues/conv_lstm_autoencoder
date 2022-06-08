from enum import Enum


BATCH_INPUT_LENGTH = 10
IMAGE_SIZE = 64  # 256

BATCH_SIZE = 4  # number of training samples per learning iteration
EPOCHS = 50  # number of times the full dataset is seen during training

TEMPORAL_STRIDE = 8
DATA_AUGMENTATION = True
seed_constant = 30

IMAGE_EXTENSION_LIST = ["tif", "jpg", "png", "jpeg"]
VIDEO_EXTENSION_LIST = ['.mp4', '.avi', '.mpg']


class eDatasets(Enum):
    UCSDPed = 1
    Turnstiles = 2

TRAIN_DATASET = eDatasets.UCSDPed
