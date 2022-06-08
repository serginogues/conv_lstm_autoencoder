from config import *
import numpy as np
from os import listdir
from os.path import join, isdir, isfile
from tqdm import tqdm
import cv2
import random


def preprocess_frame(frame):
    """
    Parameters
    ----------
    frame
        cv2 frame

    Returns
    -------
    frame
        frame with shape (IMAGE_SIZE, IMAGE_SIZE, 1) and scaled between 0 and 1
    """
    # reshape and normalize frame
    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_frame = cv2.resize(new_frame, (IMAGE_SIZE, IMAGE_SIZE)) / 256.0
    new_frame = np.reshape(new_frame, (IMAGE_SIZE, IMAGE_SIZE, 1))

    return new_frame


def create_dataset(dataset_path: str):
    """
    Parameters
    ----------
    dataset_path
        path to repo with videos
    Returns
    -------
    numpy array
        array of shape (# clips, BATCH_INPUT_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 1)
    """
    clips = []

    # iterate through video list
    for vid in sorted(listdir(dataset_path)):
        video_path = join(dataset_path, vid)
        if isfile(video_path) and (vid.endswith(VIDEO_EXTENSION_LIST[0])
                            or vid.endswith(VIDEO_EXTENSION_LIST[1])
                            or vid.endswith(VIDEO_EXTENSION_LIST[2])):
            # extract frames from video
            stride_list = [5, 10, 20] if DATA_AUGMENTATION else [TEMPORAL_STRIDE]
            for stride in stride_list:
                video_clips = extract_frames_by_stride(video_path, stride)
                clips.extend(video_clips)

    return np.asarray(clips)


def extract_frames_by_stride(path: str, stride: int) -> list:
    """
    Parameters
    ----------
    path
        path to video
    stride
        temporal stride
    Returns
    -------
    list
        list of clips, (BATCH_INPUT_SHAPE x IMG_SIZE x IMG_SIZE x C)
    """

    clip = np.zeros(shape=(BATCH_INPUT_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 1))

    # create video capture
    vidcap = cv2.VideoCapture(path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    # num_clips = int((total_frames / stride) / BATCH_INPUT_SHAPE)

    list_clips = []
    cnt = 0
    # run through all video frames
    for idx in range(total_frames):

        # read next frame
        success, frame = vidcap.read()
        if not success: break

        # do something with temporal stride
        if idx % stride == 0:
            clip[cnt, :, :, :] = preprocess_frame(frame)
            cnt += 1
            if cnt == BATCH_INPUT_LENGTH:
                list_clips.append(np.copy(clip))
                cnt = 0

    return list_clips


def visualize_frames(clip: np.ndarray, label: np.ndarray, classes: list):
    """
    Parameters
    ----------
    clip
        (BATCH_INPUT_SHAPE x IMG_SIZE x IMG_SIZE x C)
    """
    numpy_horizontal = np.hstack((clip[0], clip[2], clip[4]))
    cv2.imshow(classes[np.argmax(label)], numpy_horizontal)