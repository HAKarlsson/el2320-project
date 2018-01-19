#!/usr/bin/python
import numpy as np
import cv2
import sys
from scipy.io import loadmat
from AntTracker import *
from aux_func import get_predictor
import matplotlib.pyplot as plt


predictor = get_predictor()

tracker = sys.argv[1]


if tracker == "Sys":
    pf = SystematicAT(predictor)
elif tracker == "Para":
    pf = ParallelAT(predictor)
elif tracker == "Hard":
    pf = HardClusteringAT(predictor)
elif tracker == "Soft":
    pf = SoftClusteringAT(predictor)
elif tracker == "Multi":
    pf = MultiComponentAT(predictor)
    print MultiComponentAT(predictor)
else:
    print("No tracker selected.")
    exit()

p1 = (16, 16)
p2 = (720-16, 480-16)

ESCAPE_KEY = 27
# capture video
video_capture = cv2.VideoCapture('Datasets/ants_1m.avi')

# gts: ground truths
tmp = loadmat('Datasets/ant_dataset_gt.mat')['traj_gt']
gts = np.zeros(tmp.shape, dtype=int)
gts[:,:,0], gts[:,:,1] = tmp[:,:,1], tmp[:,:,0]

# t: time
t = 0

#
accs = np.zeros(1800)

# while video has not been released yet (or still opened)
while True:
    ret, frame = video_capture.read()
    gt_t = gts[:,t,:]
    if not ret:
        break
    # convert to gray image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    part = pf.sisr(gray).astype(int)[:, 0:2]
    frame[part[:, 0], part[:, 1], :] = [255, 0, 0]
    frame[gt_t[:, 0], gt_t[:, 1], :] = [0, 255, 0]
    accuracy = 0
    for i in range(20):
        dist = np.linalg.norm(part - gt_t[i, :], axis=1)
        if np.any(dist < 16):
            accuracy += 1
    accs[t] = accuracy
    cv2.rectangle(frame, p1, p2, (0, 0, 255))

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    t += 1
# close video stream
video_capture.release()
# as it describes, destroy all windows has been opened by opencv
cv2.destroyAllWindows()

plt.plot(range(1800), accs/20)
plt.show()