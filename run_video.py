import argparse
from config import *
import numpy as np
from keras.models import load_model
from keras.layers import LayerNormalization
import cv2
import time


def evaluate_clip(clip: list, model):
    """
    Run reconstruction on single clip and evaluate error.

    Parameters
    ----------
    clip
        list of (255, 255, 1) frames, where len(frame_list) = BATCH_INPUT_SHAPE
    model
        backup/model.hdf5
    """
    sequences = np.zeros((1, BATCH_INPUT_SHAPE, 256, 256, 1))
    for i in range(BATCH_INPUT_SHAPE):
        sequences[0, i, :, :, :] = clip[i]

    reconstructed_sequences = model.predict(sequences)
    sequences_reconstruction_cost = np.linalg.norm(np.subtract(sequences[0], reconstructed_sequences[0]))
    return sequences_reconstruction_cost


def run_video(path: str):
    """
    Evaluate video
    """
    VIDEO_PATH = path
    model = load_model(SAVE_PATH, custom_objects={'LayerNormalization': LayerNormalization})

    # begin video capture
    # if the input is the camera, pass 0 instead of the video path
    try:
        vid = cv2.VideoCapture(VIDEO_PATH)
    except:
        vid = cv2.VideoCapture(VIDEO_PATH)

    # init display params
    start = time.time()
    counter = 0

    # init video_clip for slowfast
    clip = []
    cost_history = []

    # read frame by frame until video is completed
    while vid.isOpened():

        ret, frame = vid.read()  # capture frame-by-frame
        if not ret: break
        counter += 1
        print("Frame #", counter)

        # checking video frame rate
        """
        start_time = time.time()
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        """

        # Writing FrameRate on video
        cv2.putText(frame, "# frame: " + str(counter), (50, 70), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

        # convert back to BGR
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if counter % 2 == 0:

            clip.append(gray_img)

            if len(clip) == 10:
                clip.pop(0)

            cost = evaluate_clip(clip, model)
            cost_history.append(cost)

            cv2.putText(frame, "Cost: " + str(cost), (50, 70), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

        # show frame
        cv2.imshow("Output Video", frame)

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


def main(config):
    run_video(config.path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to video')
    config = parser.parse_args()
    main(config)