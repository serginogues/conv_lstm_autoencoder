from config import *
from os import listdir
import os.path
from os.path import isfile, join, isdir
import numpy as np
from tqdm import tqdm
import PIL.Image


def get_clips_by_stride(frames_list: list, stride: int = 1):
    """
    Create list of 10-frames sequences.

    Parameters
    ----------
    stride : int
        The distance between two consecutive frames.
        To apply data augmentation set stride to 2 or 3.
        Example (1, 3, 5, 7, 9, 11, 13, 15, 17, 19).
    frames_list : list
        A list of frames of shape 256 X 256. It should be a video sequence.
    clip_size: int
        The size of the lstm input sequence
    Returns
    -------
    list
        A list of clips , 10 frames each (10 x 256 x 256 x 1)
    """
    clips = []
    clip = np.zeros(shape=(BATCH_INPUT_SHAPE, 256, 256, 1))
    img_idx = 0
    for start in range(0, stride):
        for i in tqdm(range(start, len(frames_list), stride), desc="Creating clips"):
            clip[img_idx, :, :, 0] = frames_list[i]
            img_idx = img_idx + 1
            if img_idx == BATCH_INPUT_SHAPE:
                clips.append(np.copy(clip))
                img_idx = 0
    return clips


def get_train_dataset(dataset_path: str = 'UCSDped1/Train', data_augmentation: bool = False):
    """
    Parameters
    ----------
    dataset_path
        path to dataset
    data_augmentation
        if True does data augmentation to create more clip sequences

    Returns
    -------
    list
        A list of clips , 10 frames each (10 x 256 x 256 x 1)
    """
    clips = []
    range_ = range(1, 3) if data_augmentation else range(2, 3)

    # loop over the training folders (video1, video2, ..)
    for f in sorted(listdir(dataset_path)):
        directory_path = join(dataset_path, f)
        if isdir(directory_path):
            all_frames = []
            # loop over all files in the folder
            for c in tqdm(sorted(listdir(directory_path)), desc="Loading " + str(f)):
                img_path = join(directory_path, c)
                # append if it is an image with FORMAT
                if img_path.split(".")[-1] in FORMATS_list:
                    img = PIL.Image.open(img_path).resize((256, 256))
                    img = np.array(img, dtype=np.float32) / 256.0
                    all_frames.append(img)
            # get the 10-frames sequences from the list of images after applying data augmentation
            for stride in range_:
                clips.extend(get_clips_by_stride(frames_list=all_frames, stride=stride))
    return clips


def get_single_test(path: str = 'UCSDped1/Test/Test032'):
    """
    Returns single 200 frame testing video specified by TEST_PATH

    Parameters
    ----------
    path
        UCSD dataset provides 34 testing videos

    Returns
    -------
    np.ndarray
        (200 x 256 x 256 x 1) numpy array
    """
    sz = int(len([name for name in os.listdir(path) if isfile(join(path, name)) and name.split(".")[-1] in FORMATS_list]))
    test = np.zeros(shape=(sz, 256, 256, 1))
    idx = 0
    for f in sorted(listdir(path)):
        file_path = join(path, f)
        if file_path.split(".")[-1] in FORMATS_list:  # "tif"
            img = PIL.Image.open(file_path).resize((256, 256))
            img = np.array(img, dtype=np.float32) / 256.0
            test[idx, :, :, 0] = img
            idx = idx + 1
    return test