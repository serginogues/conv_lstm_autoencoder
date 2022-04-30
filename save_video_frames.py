import argparse
from pathlib import Path
import cv2


def save_video_frames(config):
    """
    Save video frames
    """
    path = config.path
    per_200_frames = config.per_200_frames

    # get path without '.mp4'
    name = path.split(".")[0]

    # create video capture
    vidcap = cv2.VideoCapture(path)

    # create parent video folder
    Path(name).mkdir(parents=True, exist_ok=True)

    if per_200_frames:
        # create subfolders
        folder_idx = [x for x in range(0, int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), 200)]
        for x in folder_idx:
            Path(name + "/train_" + str(x)).mkdir(parents=True, exist_ok=True)

        # start reading frames
        success, image = vidcap.read()
        count = 0
        folder_count = 0
        while success:

            # new folder every 200 frames
            if count > 199:
                count = 0
                folder_count += 1
                print(str(folder_count) + " out of " + str(len(folder_idx)))

            print(count)
            # save in grayscale 0-255
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            new_path = name + "/train_" + str(folder_idx[folder_count]) + "/frame" + str(count) + ".jpg"
            cv2.imwrite(new_path, gray)

            # next frame
            success, image = vidcap.read()
            count += 1
    else:
        # start reading frames
        success, image = vidcap.read()
        count = 0
        while success:
            print(count)
            # save in grayscale 0-255
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            new_path = name + "/frame" + str(count) + ".jpg"
            cv2.imwrite(new_path, gray)

            # next frame
            success, image = vidcap.read()
            count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--per_200_frames', type=bool, default=False,
                        help='if true saves video in subfolders of 200 frames each')
    parser.add_argument('--path', type=str, default='UCSDped1/Train',
                        help='path to dataset or test video')
    config = parser.parse_args()
    save_video_frames(config)