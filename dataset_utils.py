from config import *
from os import listdir
from os.path import join, isdir, isfile
import numpy as np
from tqdm import tqdm
import PIL.Image
from keras.models import load_model
from keras.layers import LayerNormalization
import matplotlib.pyplot as plt


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
    ucsd_train_path = 'C:/Users/azken/Documents/Datasets/Anomaly Detection/UCSDped1/Train'
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


def test_UCSDPed1(model_path: str):
    """
    Parameters
    ----------
    model_path
        path to .h5 file
    """

    path = 'C:/Users/azken/Documents/Datasets/Anomaly Detection/UCSDped1/Test'
    model = load_model(model_path, custom_objects={'LayerNormalization': LayerNormalization})

    for vid in listdir(path):
        vid_path = join(path, vid)
        if isdir(vid_path):

            test = get_single_test_UCSDPed1(vid_path)
            sz = test.shape[0] - BATCH_INPUT_LENGTH
            sequences = np.zeros((sz, BATCH_INPUT_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 1))

            for i in tqdm(range(0, sz), desc="Creating clips"):
                clip = np.zeros((BATCH_INPUT_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 1))
                for j in range(0, BATCH_INPUT_LENGTH):
                    clip[j] = test[i + j, :, :, :]
                    # reconstructed_sequence = model.predict(sequences, batch_size=1, verbose=1)
                sequences[i] = clip

            # get the reconstruction cost of all the sequences
            reconstructed_sequences = model.predict(sequences, batch_size=BATCH_SIZE, verbose=1)
            sequences_reconstruction_cost = np.array(
                [np.linalg.norm(np.subtract(sequences[i], reconstructed_sequences[i])) for i in range(0, sz)])

            # abnormality score
            sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(
                sequences_reconstruction_cost)

            # regularity score
            sr = 1.0 - sa

            # plot the regularity scores
            plt.plot(sr, label=vid)
            plt.legend()
            plt.ylabel('Regularity Score')
            plt.xlabel('Frame')
            plt.show()


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
    test = np.zeros(shape=(int(len(frame_path_list)), IMAGE_SIZE, IMAGE_SIZE, 1))
    idx = 0
    for img_path in frame_path_list:
        img = PIL.Image.open(img_path).resize((IMAGE_SIZE, IMAGE_SIZE))
        img = np.array(img, dtype=np.float32) / 256.0
        test[idx, :, :, 0] = img
        idx = idx + 1
    return test


test_UCSDPed1('backup/UCSDPed_0.0016_1958209_15_10_64.hdf5')
