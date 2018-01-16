import numpy as np
from sklearn.neural_network import MLPClassifier


def get_predictor():
    data = np.load('Datasets/ant32.npy')
    X = normImg(data[:, :1024])
    y = data[:, 1024]
    layers = (32,)*2
    predictor = MLPClassifier(
        activation='relu', solver='adam', hidden_layer_sizes=layers)
    predictor.fit(X, y)
    return predictor


def normImg(img):
    # set image values to range [-1, 1]
    # for efficient prediction
    return (img.astype(float) - 128.) / 256.
