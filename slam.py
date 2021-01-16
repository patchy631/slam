import cv2
import time
import numpy as np
from display import Display
from extrcator import Extractor

W = 1920//2
H = 1080//2

disp = Display(W, H)

fe = Extractor()

def process_frame(img):
    img = cv2.resize(img, (W, H))
    matches = fe.extract(img)
    if matches is None:
        return
    for pt1, pt2 in matches:
        u1, v1 = map(lambda x: int(round(x)), pt1)
        u2, v2 = map(lambda x: int(round(x)), pt2)
        cv2.circle(img, center=(u1, v1), radius=3, color=(0, 255, 0))
        cv2.line(img, (u1, v1), (u2, v2), color=(0, 0, 255))

    disp.paint(img)


if __name__ == '__main__':
    cap = cv2.VideoCapture("test.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            process_frame(frame)
        else:
            break