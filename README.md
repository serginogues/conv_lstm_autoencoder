# LSTM Convolutional Autoencoder for Anomaly Detection in Videos
Spatio-temporal autoencoder to reconstruct frame sequences.

## Train a new model
Edit config file ``config.py`` and
run ``python train.py --path [path to training dataset]``

## Evaluate trained model
Run ``python evaluate.py --path [path to folder with single video frames]``

## Real-time inference
Run ``python run_video.py --path [path to video]``

## References
- Yong Shean Chong, Abnormal Event Detection in Videos using Spatiotemporal Autoencoder (2017) https://arxiv.org/abs/1701.01546
- Mahmudul Hasan, Jonghyun Choi, Jan Neumann, Amit K. Roy-Chowdhury, Learning Temporal Regularity in Video Sequences (2016) https://arxiv.org/abs/1604.04574
- UCSD Anomaly Detection Dataset: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm
- Hashem Sellat, Anomaly Detection in Videos using LSTM Convolutional Autoencoder (2019): https://towardsdatascience.com/prototyping-an-anomaly-detection-system-for-videos-step-by-step-using-lstm-convolutional-4e06b7dcdd29
