#!/usr/bin/python
import numpy as np
import cv2
import sys
from scipy.io import loadmat
from AntTracker import *
from aux_func import get_predictor
import matplotlib.pyplot as plt

def getGroundTruth():
    # gts: ground truths
    tmp = loadmat('Datasets/ant_dataset_gt.mat')['traj_gt']
    gts = np.zeros(tmp.shape, dtype=int)
    gts[:,:,0], gts[:,:,1] = tmp[:,:,1], tmp[:,:,0]
    return gts

def particleAccuracy(gt, part):
    accuracy = 0.
    for i in range(20):
        dist = np.linalg.norm(part - gt[i, :], axis=1)
        if np.any(dist < 16):
            accuracy += 1
    return accuracy / 20

def getFrames(T):
    video_capture = cv2.VideoCapture('Datasets/ants_1m.avi')
    frames = list()
    for t in range(T):
        ret, frame = video_capture.read()
        # convert to gray image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    video_capture.release()
    return frames

def calcAccuracy(tracker, predictor, T = 1799):
    # accuracy matrix
    accs = np.zeros((10, T))
    # ground truths
    gts= getGroundTruth()
    # video frames
    frames = getFrames(T)

    for i in range(10):
        # Restart particle filter
        pf = pfc(predictor)
        for t, frame in enumerate(frames):
            gt_t = gts[:,t,:]
            part = pf.sisr(frame).astype(int)[:, 0:2]
            accs[i, t] = particleAccuracy(gt_t, part)
    return np.mean(accs, 0), np.std(accs,0)


tracker = sys.argv[1]
predictor = get_predictor()
if tracker == "Sys":
    pfc = SystematicAT
    title = "Systematic Resampling"
    saveas = "../figs/sys"
elif tracker == "Para":
    title = "Parallel Resampling"
    pfc = ParallelAT
    saveas = "../figs/para2"
elif tracker == "Hard":
    pfc = HardClusteringAT
    title = "K-means Resampling (Lloyd's)"
    saveas = "../figs/hard"
elif tracker == "Soft":
    pfc = SoftClusteringAT
    title = "K-means Resampling (EM)"
    saveas = "../figs/soft"

T = 1799

accMean, accStd = calcAccuracy(pfc, predictor, T)
plt.fill_between(range(1, T+1), accMean-accStd, accMean+accStd, facecolor='green', alpha=.5)
plt.plot(range(1, T+1), accMean, 'g')
plt.title(title)
plt.xlabel('Avg. Accuracy')
plt.ylabel('Time')
plt.axis((1, T+1, 0, 1))
plt.savefig(saveas)
plt.show()