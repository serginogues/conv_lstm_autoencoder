from os import listdir
from os.path import isfile, join, isdir
import numpy as np
from tqdm import tqdm
import PIL.Image


def get_clips_by_stride(stride, frames_list, sequence_size):
    """ Data augmentation in the temporal dimension. \n
    Example: First stride made of frames (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    Second stride made of frames (1, 3, 5, 7, 9, 11, 13, 15, 17, 19). \n
    Parameters
    ----------
    stride : int
        The distance between two consecutive frames
    frames_list : list
        A list of sorted frames of shape 256 X 256
    sequence_size: int
        The size of the lstm sequence
    Returns
    -------
    list
        A list of clips , 10 frames each (256 x 256 x 1)
    """
    clips = []
    sz = len(frames_list)
    clip = np.zeros(shape=(sequence_size, 256, 256, 1))
    cnt = 0
    for start in range(0, stride):
        for i in range(start, sz, stride):
            clip[cnt, :, :, 0] = frames_list[i]
            cnt = cnt + 1
            if cnt == sequence_size:
                clips.append(np.copy(clip))
                cnt = 0
    return clips


def get_train_ucsd(dataset_path: str = 'UCSDped1/Train'):
    """
    Returns
    -------
    list
        A list of training sequences of shape (NUMBER_OF_SEQUENCES,SINGLE_SEQUENCE_SIZE,FRAME_WIDTH,FRAME_HEIGHT,1)
    """
    clips = []
    # loop over the training folders (Train000,Train001,..)
    for f in tqdm(sorted(listdir(dataset_path)), desc="Loading Dataset"):
        directory_path = join(dataset_path, f)
        if isdir(directory_path):
            all_frames = []
            # loop over all the images in the folder (0.tif,1.tif,..,199.tif)
            for c in sorted(listdir(directory_path)):
                img_path = join(directory_path, c)
                if str(img_path)[-3:] == "tif":
                    img = PIL.Image.open(img_path).resize((256, 256))
                    img = np.array(img, dtype=np.float32) / 256.0
                    all_frames.append(img)
            # get the 10-frames sequences from the list of images after applying data augmentation
            for stride in range(1, 3):
                clips.extend(get_clips_by_stride(stride=stride, frames_list=all_frames, sequence_size=10))
    return clips


def get_single_test(TEST_PATH: str = 'UCSDped1/Test/Test032'):
    """
    Returns single 200 frame testing video specified by TEST_PATH. \n
    UCSD dataset provides 34 testing videos. \n
    :returns: (200 x 256 x 256 x 1) numpy array
    """
    sz = 200
    test = np.zeros(shape=(sz, 256, 256, 1))
    idx = 0
    for f in sorted(listdir(TEST_PATH)):
        if str(join(TEST_PATH, f))[-3:] == "tif":
            img = PIL.Image.open(join(TEST_PATH, f)).resize((256, 256))
            img = np.array(img, dtype=np.float32) / 256.0
            test[idx, :, :, 0] = img
            idx = idx + 1
    return test