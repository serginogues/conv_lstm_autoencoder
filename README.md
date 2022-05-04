# LSTM Convolutional Autoencoder for Anomaly Detection in Videos
Spatio-temporal autoencoder to reconstruct frame sequences.

## Train a new model
Edit config file ``config.py`` and
run ``python train.py --path [path to training dataset]``

## Evaluate trained model
Run ``python evaluate.py --path [path to folder with single video frames]``

## Real-time inference
Run ``python run_video.py --path [path to video]``
