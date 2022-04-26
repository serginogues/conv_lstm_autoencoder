"""
LSTM Convolutional Autoencoder for Anomaly Detection in Videos
UCSD Anomaly Detection Dataset: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm
https://towardsdatascience.com/prototyping-an-anomaly-detection-system-for-videos-step-by-step-using-lstm-convolutional-4e06b7dcdd29
"""

from autoencoder import get_model

if __name__ == '__main__':
    get_model(False)