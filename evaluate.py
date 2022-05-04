import argparse
from config import *
import numpy as np
from keras.models import load_model
from keras.layers import LayerNormalization
import matplotlib.pyplot as plt
from dataset_utils import get_single_test
from video2frames import video2frames
from tqdm import tqdm


def evaluate_frame_sequence(path: str):
    """
    Evaluate single video already stored as frame sequence
    """
    print(SAVE_PATH)
    model = load_model(SAVE_PATH, custom_objects={'LayerNormalization': LayerNormalization})
    print("Model used: " + SAVE_PATH)
    test = get_single_test(path=path)

    sz = test.shape[0] - BATCH_INPUT_SHAPE
    sequences = np.zeros((sz, BATCH_INPUT_SHAPE, 256, 256, 1))

    for i in tqdm(range(0, sz), desc="Creating clips"):
        clip = np.zeros((BATCH_INPUT_SHAPE, 256, 256, 1))
        for j in range(0, BATCH_INPUT_SHAPE):
            clip[j] = test[i + j, :, :, :]
            # reconstructed_sequence = model.predict(sequences, batch_size=1, verbose=1)
        sequences[i] = clip

    # get the reconstruction cost of all the sequences
    reconstructed_sequences = model.predict(sequences, batch_size=1 , verbose=1)
    sequences_reconstruction_cost = np.array(
        [np.linalg.norm(np.subtract(sequences[i], reconstructed_sequences[i])) for i in range(0, sz)])
    """sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(sequences_reconstruction_cost)
    sr = 1.0 - sa"""

    # plot the regularity scores
    plt.plot(sequences_reconstruction_cost)
    plt.ylabel('Reconstruction Cost')
    plt.xlabel('Frame')
    plt.show()


def main(config):
    path = config.path
    if config.video2frames:
        path = video2frames(path)
    evaluate_frame_sequence(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='UCSDped1/Test/Test032',
                        help='path to video frames repository')
    parser.add_argument('--video2frames', type=bool, default=False,
                        help='path to video frames repository')
    config = parser.parse_args()
    main(config)