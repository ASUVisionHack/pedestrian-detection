import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt

def render(frame):
    frame = imutils.resize(frame, height = 500)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5, 5), 1000)
    # frame = frame[400:500, 0:1920]

    edged = cv2.Canny(frame, 15, 30)
    ret, edged = cv2.threshold(edged,127,255,1)
    image, contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow('img',image)
    cv2.waitKey(0)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.06 * cv2.arcLength(cnt,True),True)
        cv2.drawContours(frame, [approx], -1, (255, 0, 0), cv2.LINE_4, 8)
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        count = len(approx)

        print('x: {:f}, y: {:f}, w: {:f}, h: {:f}, ar: {:f}'.format(x, y, w, h, ar))
        # print('epsilon: {:f}, num: {:f}'.format(epsilon, count))

    cv2.imshow('img',frame)
    cv2.waitKey(0)


image = cv2.imread('sample-img-1.png')
render(image)
cv2.destroyAllWindows()
