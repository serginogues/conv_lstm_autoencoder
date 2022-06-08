from config import *
from os import listdir
from os.path import join, isdir, isfile
import numpy as np
from tqdm import tqdm
import PIL.Image


def create_UCSDPed1():
    """
    Parameters
    ----------
    Returns
    -------
    np.array
        numpy array of shape (# clips, BATCH_INPUT_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 1)
    """
    dataset = []
    ucsd_train_path = 'data/UCSDped1/Train'
    # loop over the training folders (video1, video2, ..)
    for f in sorted(listdir(ucsd_train_path)):
        directory_path = join(ucsd_train_path, f)
        if isdir(directory_path):
            # loop over all frames in the folder
            all_frames = []
            for c in tqdm(sorted(listdir(directory_path)), desc="Loading " + str(f)):
                img_path = join(directory_path, c)
                # append if it is an image
                if img_path.split(".")[-1] in IMAGE_EXTENSION_LIST:
                    img = PIL.Image.open(img_path).resize((IMAGE_SIZE, IMAGE_SIZE))
                    img = np.array(img, dtype=np.float32) / 256.0
                    all_frames.append(img)

            # create clips and store in dataset
            stride_list = [5, 10, 20] if DATA_AUGMENTATION else [TEMPORAL_STRIDE]
            clips = []
            for stride in stride_list:

                """for start_frame_idx in [0, int(len(all_frames) * 0.25), int(len(all_frames) * 0.5),
                                        int(len(all_frames) * 0.75)]:"""
                clip = np.zeros(shape=(BATCH_INPUT_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 1))
                img_idx = 0
                for i in range(0, len(all_frames), stride):
                    clip[img_idx, :, :, 0] = all_frames[i]
                    img_idx = img_idx + 1
                    if img_idx == BATCH_INPUT_LENGTH:
                        clips.append(np.copy(clip))
                        img_idx = 0
            dataset.extend(clips)
    return np.asarray(dataset)


def get_single_test_UCSDPed1(path: str):
    """
    Parameters
    ----------
    path
        Path to single test folder containing video frames (images)
    Returns
    -------
    np.ndarray
        (#images x 256 x 256 x 1) numpy array
    """
    frame_path_list = [join(path, name) for name in listdir(path) if isfile(join(path, name)) and name.split(".")[-1] in IMAGE_EXTENSION_LIST]
    test = np.zeros(shape=(int(len(frame_path_list)), 256, 256, 1))
    idx = 0
    for img_path in frame_path_list:
        img = PIL.Image.open(img_path).resize((256, 256))
        img = np.array(img, dtype=np.float32) / 256.0
        test[idx, :, :, 0] = img
        idx = idx + 1
    return test
