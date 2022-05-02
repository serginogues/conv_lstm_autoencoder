"""
LSTM Convolutional Autoencoder for Anomaly Detection in Videos
UCSD Anomaly Detection Dataset: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm
https://towardsdatascience.com/prototyping-an-anomaly-detection-system-for-videos-step-by-step-using-lstm-convolutional-4e06b7dcdd29

---------------------------------------------------------------------
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

Why Conv-LSTM?
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
from config import *
import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.layers import Conv2DTranspose, ConvLSTM2D, TimeDistributed, Conv2D, LayerNormalization
from keras.models import Sequential, load_model
from dataset_utils import get_train_dataset, get_single_test
import tensorflow as tf
import time


def train(path):
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
    training_set = get_train_dataset(dataset_path=path)

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

    return seq


def evaluate_frame_sequence(path: str = 'UCSDped1/Test/Test032'):
    """
    Evaluate single video already stored as frame sequence
    """
    model = load_model(SAVE_PATH, custom_objects={'LayerNormalization': LayerNormalization})
    test = get_single_test(path=path)
    sz = test.shape[0] - BATCH_INPUT_SHAPE
    sequences = np.zeros((sz, BATCH_INPUT_SHAPE, 256, 256, 1))

    for i in range(0, sz):
        clip = np.zeros((BATCH_INPUT_SHAPE, 256, 256, 1))
        for j in range(0, BATCH_INPUT_SHAPE):
            clip[j] = test[i + j, :, :, :]
        sequences[i] = clip

    # get the reconstruction cost of all the sequences
    reconstructed_sequences = model.predict(sequences, batch_size=4)
    sequences_reconstruction_cost = np.array(
        [np.linalg.norm(np.subtract(sequences[i], reconstructed_sequences[i])) for i in range(0, sz)])
    sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(sequences_reconstruction_cost)
    sr = 1.0 - sa

    # plot the regularity scores
    plt.plot(sr)
    plt.ylabel('regularity score Sr(t)')
    plt.xlabel('frame t')
    plt.show()


def evaluate_video(config):
    """
    Evaluate video
    """
    SAVE = False
    VIDEO_PATH = config.path

    # begin video capture
    # if the input is the camera, pass 0 instead of the video path
    try:
        vid = cv2.VideoCapture(VIDEO_PATH)
    except:
        vid = cv2.VideoCapture(VIDEO_PATH)

    # get video ready to save locally if flag is set
    out = None
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if SAVE:
        # by default VideoCapture returns float instead of int
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*"mp4")  # 'XVID'
        out = cv2.VideoWriter("output", codec, fps, (frame_width, frame_height))

    # init display params
    start = time.time()
    counter = 0

    # init video_clip for slowfast
    video_clip = []

    # read frame by frame until video is completed
    while vid.isOpened():

        ret, frame = vid.read()  # capture frame-by-frame
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            video_clip.append(frame)
        else:
            print('Video has ended or failed')
            break

        start_time = time.time()
        counter += 1
        print("Frame #", counter)

        # Display
        # checking video frame rate
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)

        # Writing FrameRate on video
        cv2.putText(frame, str(int(fps)) + " fps", (50, 70), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

        # convert back to BGR
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # if flag save video, else display
        if SAVE:
            out.write(result)
        else:
            # show frame
            cv2.imshow("Output Video", result)

            # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Closes all the frames
    cv2.destroyAllWindows()

    # Average fps
    end = time.time()
    seconds = end - start
    print("Time taken: {0} seconds".format(seconds))
    print("Number of frames: {0}".format(counter))
    fps = counter / seconds
    print("Estimated frames per second: {0}".format(fps))

    # When everything done, release the video capture object
    vid.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=False,
                        help='Test saved model or retrain it')
    parser.add_argument('--path', type=str, default='UCSDped1/Train',
                        help='path to dataset or test video')
    config = parser.parse_args()

    path = config.path
    if config.train:
        train(path)
    else:
        evaluate_video(config)
