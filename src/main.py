#!/usr/bin/python
import numpy as np
import cv2
from AntTracker import *
from aux_func import get_predictor


predictor = get_predictor()

tracker = 1


if tracker == 0:
    pf = SystematicAT(predictor)
elif tracker == 1:
    pf = KMeansAT(predictor)

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
