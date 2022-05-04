"""
Spatio-Temporal Autoencoder with LSTM and Convolutional layers.

Architecture:
- Input: (10 x 256 x 256 x 1) 10 grayscale images that form a sequence/clip

################ Encoder ############################################
-------------- Spatial encoder --------------------------------------
- Conv: 11 x 11 x 128 filters with stride 4 -> (10 x 64 x 64 x 128)
- Conv: 5 x 5 x 64 filters with stride 2    -> (10 x 32 x 32 x 64)

-------------- Temporal encoder -------------------------------------
- ConvLSTM2D: 3 x 3 x 64 filters -> (10 x 32 x 32 x 64)
- ConvLSTM2D: 3 x 3 x 32 filters -> (10 x 32 x 32 x 32) latent space

################ Decoder ############################################
-------------- Temporal decoder -------------------------------------
- ConvLSTM2D: 3 x 3 x 64 filters -> (10 x 32 x 32 x 64)

-------------- Spatial decoder --------------------------------------
- Deconv: 5 x 5 x 64 filters with stride 2 -> (10 x 64 x 64 x 64)
- Deconv: 11 x 11 x 128 filters with stride 2 -> (10 x 256 x 256 x 128)
- Conv: 11 x 11 x 1 filter -> (10 x 256 x 256 x 1)
- Output: reconstruction of input

About keras.layers.TimeDistributed:
    Consider a batch of 32 video samples, where each sample is a 128x128 RGB image with channels_last data format,
    across 10 timesteps. The batch input shape is (32, 10, 128, 128, 3).
    You can then use TimeDistributed to apply the same Conv2D layer to each of the 10 timesteps, independently.
    Because TimeDistributed applies the same instance of Conv2D to each of the timestamps,
    the same set of weights are used at each timestamp.
"""
from config import *
import argparse
import numpy as np
from keras.layers import Conv2DTranspose, ConvLSTM2D, TimeDistributed, Conv2D, LayerNormalization
from keras.models import Sequential
from dataset_utils import get_train_dataset
import tensorflow as tf


def train(config):
    """
    Train conv-lstm autoencoder

    Parameters
    ----------
    path
        path to training dataset

    Returns
    -------
    weights
        saves model at backup/model.hdf5
    """
    # get taining data
    training_set = get_train_dataset(dataset_path=config.path)

    # training set is an array of clips with shape (# img, 10, 256, 256, 1)
    training_set = np.array(training_set)

    seq = Sequential()

    # Spatial encoder
    seq.add(TimeDistributed(Conv2D(128, (11, 11), strides=4, padding="same"), batch_input_shape=(None, BATCH_INPUT_SHAPE, 256, 256, 1)))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2D(64, (5, 5), strides=2, padding="same")))
    seq.add(LayerNormalization())

    # Temporal encoder
    seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    seq.add(ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())

    # Temporal decoder
    seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())

    # Spatial decoder
    seq.add(TimeDistributed(Conv2DTranspose(64, (5, 5), strides=2, padding="same")))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2DTranspose(128, (11, 11), strides=4, padding="same")))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2D(1, (11, 11), activation="sigmoid", padding="same")))

    print(seq.summary())
    print("############# Training set shape: " + str(training_set.shape))

    # compile model architecture
    seq.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5, epsilon=1e-6))

    # train our model
    print("Training starts")
    seq.fit(training_set, training_set, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False, verbose=1)
    seq.save(SAVE_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='UCSDped1/Train',
                        help='path to dataset or test video')
    config = parser.parse_args()
    train(config)
