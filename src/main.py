#!/usr/bin/python
import numpy as np
import cv2
<<<<<<< HEAD
import sys
=======
>>>>>>> d010e1e11202591ae857699e4d7004aa79574d3d
from AntTracker import *
from aux_func import get_predictor


predictor = get_predictor()

<<<<<<< HEAD
tracker = sys.argv[1]


if tracker == "Sys":
    pf = SystematicAT(predictor)
elif tracker == "Para":
    pf = ParallelAT(predictor)
elif tracker == "Hard":
    pf = HardClusteringAT(predictor)
elif tracker == "Soft":
    pf = SoftClusteringAT(predictor)
else:
    print("No tracker selected.")
    exit()
=======
tracker = 1


if tracker == 0:
    pf = SystematicAT(predictor)
elif tracker == 1:
    pf = KMeansAT(predictor)
>>>>>>> d010e1e11202591ae857699e4d7004aa79574d3d

p1 = (16, 16)
p2 = (720-16, 480-16)

ESCAPE_KEY = 27
# capture video
video_capture = cv2.VideoCapture('Datasets/ants_1m.avi')
# while video has not been released yet (or still opened)
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    # convert to gray image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    part = pf.sisr(gray).astype(int)
    frame[part[:, 0], part[:, 1], :] = [0, 255, 0]
    cv2.rectangle(frame, p1, p2, (0, 0, 255))

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close video stream
video_capture.release()
# as it describes, destroy all windows has been opened by opencv
cv2.destroyAllWindows()
