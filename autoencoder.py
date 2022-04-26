"""
Spatio-Temporal Autoencoder with LSTM and Convolutional layers
Encoder made of two parts: spatial encoder and temporal encoder.

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

Why Conv LSTM?
    We used convolutional LSTM layers instead of fully connected LSTM layers
    because FC-LSTM layers do not keep the spatial data very well because of its usage of full connections in input-to-state
    and state-to-state transitions in which no spatial information is encoded.

About keras.layers.TimeDistributed:
    Consider a batch of 32 video samples, where each sample is a 128x128 RGB image with channels_last data format,
    across 10 timesteps. The batch input shape is (32, 10, 128, 128, 3).
    You can then use TimeDistributed to apply the same Conv2D layer to each of the 10 timesteps, independently.
    Because TimeDistributed applies the same instance of Conv2D to each of the timestamps,
    the same set of weights are used at each timestamp.
"""
from os import listdir
from os.path import join

import numpy as np
import keras
import matplotlib.pyplot as plt
from PIL.Image import Image
from keras.layers import Conv2DTranspose, ConvLSTM2D, TimeDistributed, Conv2D, LayerNormalization
from keras.models import Sequential, load_model
from load_data import get_train_ucsd
import tensorflow as tf

SAVE_PATH = 'backup/model.hdf5'
DATASET_PATH = 'data/UCSDped1'
BATCH_SIZE = 4  # number of training samples per learning iteration
EPOCHS = 3  # number of times the full dataset is seen during training
TEST_PATH = 'data/UCSDped1/Test/Test032'


def get_model(reload_model=False):
    """
    Parameters
    ----------
    reload_model : bool
        Load saved model or retrain it
    """
    if reload_model:
        return load_model(SAVE_PATH, custom_objects={'LayerNormalization': LayerNormalization})

    # get taining data
    training_set = get_train_ucsd()
    training_set = np.array(training_set)

    seq = Sequential()

    # Spatial encoder
    seq.add(TimeDistributed(Conv2D(128, (11, 11), strides=4, padding="same"), batch_input_shape=(None, 10, 256, 256, 1)))
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

    # compile model architecture
    seq.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, decay=1e-5, epsilon=1e-6))

    # train our model
    print("Training starts")
    seq.fit(training_set, training_set, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False, verbose=2)
    seq.save(SAVE_PATH)
    return seq


def get_single_test():
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
            img = Image.open(join(TEST_PATH, f)).resize((256, 256))
            img = np.array(img, dtype=np.float32) / 256.0
            test[idx, :, :, 0] = img
            idx = idx + 1
    return test


def evaluate():
    """
    Each testing video has 200 frames.
    Use sliding window to get all the consecutive 10-frames sequences.
    :return:
    """
    model = get_model(reload_model=True)
    test = get_single_test()
    sz = test.shape[0] - 10
    sequences = np.zeros((sz, 10, 256, 256, 1))

    for i in range(0, sz):
        clip = np.zeros((10, 256, 256, 1))
        for j in range(0, 10):
            clip[j] = test[i + j, :, :, :]
        sequences[i] = clip

    # get the reconstruction cost of all the sequences
    reconstructed_sequences = model.predict(sequences,batch_size=4)
    sequences_reconstruction_cost = np.array([np.linalg.norm(np.subtract(sequences[i], reconstructed_sequences[i])) for i in range(0, sz)])
    sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(sequences_reconstruction_cost)
    sr = 1.0 - sa

    # plot the regularity scores
    plt.plot(sr)
    plt.ylabel('regularity score Sr(t)')
    plt.xlabel('frame t')
    plt.show()