#!/usr/bin/python
import numpy as np
from numpy.random import randint
import cv2
from antPF2 import anty2
from scipy.cluster.hierarchy import fclusterdata
from scipy.io import loadmat
d = 32
D = d**2


def get_ants(gt_t, gray):
    # we have 20 ants
    X = np.zeros((20, D))
    for i in range(20):
        # get ant samples
        x = gt_t[i, 0] - d / 2
        y = gt_t[i, 1] - d / 2
        X[i, :] = gray[y:y+d, x:x+d].flatten()
    return X


def get_other(gt_t, gray):
    # get 40 images where there are no ants
    X = np.zeros((160, D))
    for i in range(160):
        # gen random point
        while True:
            x, y = randint(0, 720 - d), randint(0, 480 - d)
            cord = np.array([x + d/2., y + d/.2])
            # check distance to ants
            dist = np.sqrt(np.sum((cord - gt_t)**2, 1))
            if np.all(dist > 10):
                X[i, :] = gray[y:y+d, x:x+d].flatten()
                break
    return X

ESCAPE_KEY = 27
# capture video
video_capture = cv2.VideoCapture('Datasets/ants_3m.avi')

# ground truth
gt = loadmat('Datasets/ant_dataset_gt.mat')['traj_gt']
gt = gt.astype(int)

ants = np.zeros((0, D))
other = np.zeros((0, D))
time = 0
period = 0
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    if time % 200 == 0:
        gt_t = gt[:, time, :]
        # change image to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # get new samples
        newAnts = get_ants(gt_t, gray)
        ants = np.vstack((ants, newAnts))
        newOther = get_other(gt_t, gray)
        other = np.vstack((other, newOther))
    time += 1


# close video stream
video_capture.release()

X = np.vstack((ants, other))
M = ants.shape[0]
Y = np.zeros((X.shape[0], 1))
Y[:M, :] = 1

print "dataset"
print "size", X.shape[0]
print "ants", M

Data = np.hstack((X, Y))
np.save('Datasets/ant%d.npy' % (d), Data)
