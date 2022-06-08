from config import *
from keras.utils.layer_utils import count_params
import numpy as np
from dataset_utils import create_UCSDPed1, test_UCSDPed1
import tensorflow as tf
import matplotlib.pyplot as plt
from network import create_ConvLSTMAutoencoder
import random

np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)


def plot_history(train_history, metric_1: str, metric_2: str):
    M1 = train_history.history[metric_1]
    M2 = train_history.history[metric_2]

    epochs = range(len(M1))

    plt.plot(epochs, M1, 'blue', label=metric_1)
    plt.plot(epochs, M2, 'red', label=metric_2)
    plt.legend()
    plt.show()


def train(display: bool = True):
    """
    Train conv-lstm autoencoder

    Parameters
    ----------
    display
        plot train history after training

    Returns
    -------
    weights
        saves model at backup/
    """
    # get taining data
    if TRAIN_DATASET == eDatasets.UCSDPed:
        training_set = create_UCSDPed1()
    else:
        training_set = create_UCSDPed1()

    print("Training set: " + str(training_set.shape))

    model = create_ConvLSTMAutoencoder()

    # compile model architecture
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5, epsilon=1e-6))

    # train our model
    print("Training starts")
    train_hist = model.fit(training_set, training_set, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, verbose=1)

    LOGS = [str(TRAIN_DATASET).split(".")[1],
            str(np.round(train_hist.history['loss'][-1], 4)),
            str(np.round(train_hist.history['accuracy'][-1], 4)),
            str(count_params(model.trainable_weights)),
            str(len(train_hist.history['loss'])),
            str(BATCH_SIZE),
            str(BATCH_INPUT_LENGTH),
            str(IMAGE_SIZE)]

    model_name = '_'.join([str(x) for x in LOGS]) + '.hdf5'
    model_path = 'backup/' + model_name
    model.save(model_path)
    print(model_name)

    if display:
        plot_history(train_history=train_hist, metric_1='loss', metric_2='accuracy')

    # EVALUATE
    if TRAIN_DATASET == eDatasets.UCSDPed:
        test_UCSDPed1(model_path=model_path)


if __name__ == '__main__':
    train()
