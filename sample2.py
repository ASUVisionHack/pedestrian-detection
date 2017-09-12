import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt

def render(image):
    image = imutils.resize(image, height = 500)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.GaussianBlur(image, (1, 1), 1000)

    subframe = image[400:410, 0:1920]

    cv2.imshow('window', image)
    cv2.imshow('subframe', subframe)


    subframe = cv2.Canny(subframe, 8, 50)
    cv2.imshow('subframe', subframe)
    cv2.waitKey(0)

    ret,thresh = cv2.threshold(subframe,127,255,0)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        epsilon = .02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        cv2.drawContours(subframe, [approx], -1, (255, 255, 0), cv2.LINE_4, 8)

    cv2.imshow('subframe', subframe)
    cv2.waitKey(0)

image = cv2.imread('sample-img-2.png')
render(image)
