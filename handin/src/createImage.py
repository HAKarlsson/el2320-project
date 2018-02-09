#!/usr/bin/python
"""
Author: Henrik Karlsson
Create an image of particles for the report
"""
import numpy as np
import cv2
import sys
from scipy.io import loadmat
from AntTracker import *
from aux_func import get_predictor
import matplotlib.pyplot as plt

def getFrames(T):
    video_capture = cv2.VideoCapture('Datasets/ants_1m.avi')
    frames = list()
    grays = list()
    for t in range(T):
        ret, frame = video_capture.read()
        # convert to gray image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
        grays.append(gray)
    video_capture.release()
    return frames, grays

def getImage(tracker, T, args):
    """
    tracker: the tracker class
    T: the video frame to save
    args: argument to tracker
    """
    # video frames
    frames, grays = getFrames(T)
    for i in range(10):
        # Restart particle filter
        pf = pfc(*args)
        for t, gray in enumerate(grays):
            pf.sisr(gray)
    frame = frames[T-1]
    part = pf.S.astype(int)[:, 0:2]
    for i in range(part.shape[0]):
        center = (part[i, 1], part[i, 0])
        cv2.circle(frame, center, 2, (0,255,0))
    return frame


tracker = sys.argv[1]
predictor = get_predictor()
args = (predictor,)
if tracker == "Sys":
    pfc = SystematicAT
    title = "Systematic Resampling"
    saveas = "../figs/sysImg"
    args += (400 ,)
elif tracker == "Para":
    title = "Parallel Resampling"
    pfc = ParallelAT
    filtNum = sys.argv[2]
    saveas = "../figs/paraImg" + filtNum
    args += (400, int(filtNum))
elif tracker == "Hard":
    title = "K-means Resampling (Lloyd's)"
    pfc = HardClusteringAT
    clusters = sys.argv[2]
    saveas = "../figs/hardImg" + clusters
    args += (400, int(clusters))
elif tracker == "Soft":
    title = "K-means Resampling (EM)"
    pfc = SoftClusteringAT
    clusters = sys.argv[2]
    saveas = "../figs/softImg" + clusters
    args += (400, int(clusters))

T = 300

frame = getImage(pfc, T, args)
cv2.imwrite(saveas + ".jpeg", frame)