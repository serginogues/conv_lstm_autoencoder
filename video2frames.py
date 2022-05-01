import argparse
from os import listdir
from os.path import join, isfile
from pathlib import Path
import cv2
from tqdm import tqdm


def video2frames(path: str):
    """
    Save video frames

    Parameters
    ----------
    path
        path to video
    """

    # get path without '.mp4'
    name = path.split(".")[0]

    # create video capture
    vidcap = cv2.VideoCapture(path)

    # create parent video folder
    Path(name).mkdir(parents=True, exist_ok=True)

    # start reading frames
    success, image = vidcap.read()
    count = 0
    while success:
        # save in grayscale 0-255
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        new_path = name + "/frame" + str(count) + ".jpg"
        cv2.imwrite(new_path, gray)

        # next frame
        success, image = vidcap.read()
        count += 1


def main(config):
    """
    Creates one folder per video, each folder containing the video frames

    Parameters
    ----------
    path
        path to folder with one or many videos
    """
    path = config.path
    for f in tqdm(sorted(listdir(path)), desc="Saving frames for each video"):
        video_path = join(path, f)
        if isfile(video_path) and video_path.split(".")[-1] == "mp4":
            video2frames(video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str,
                        help='path to folder with one or many videos. Creates one folder per video.')
    config = parser.parse_args()
    main(config)

