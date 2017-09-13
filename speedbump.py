import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
import os
import math

data_info = {
    "akn.022.008.left.avi": [0, 0, 0, 1, 0, 0],
    "akn.031.029.left.avi": [0, 0, 0, 1, 0, 1],
    "akn.031.037.left.avi": [0, 0, 0, 1, 1, 1],
    "akn.078.031.left.avi": [0, 0, 1, 0, 0, 0],
    "akn.082.013.left.avi": [0, 0, 1, 0, 0, 0],
    "akn.082.021.left.avi": [0, 0, 1, 0, 0, 0],
    "akn.083.014.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.083.044.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.084.048.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.088.141.left.avi": [0, 0, 0, 1, 0, 1],
    "akn.090.144.left.avi": [0, 0, 0, 1, 0, 0],
    "akn.098.056.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.113.165.left.avi": [0, 0, 0, 1, 0, 0],
    "akn.119.217.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.119.266.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.131.066.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.131.068.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.131.123.left.avi": [0, 0, 0, 1, 0, 0],
    "akn.136.026.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.136.122.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.144.212.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.159.018.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.160.025.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.160.032.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.160.036.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.160.225.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.163.014.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.163.175.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.165.120.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.166.144.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.174.010.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.174.030.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.174.056.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.174.083.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.174.086.left.avi": [0, 0, 0, 0, 1, 1],
    "akn.174.090.left.avi": [0, 0, 0, 0, 1, 1],
    "akn.174.096.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.174.100.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.174.106.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.174.124.left.avi": [0, 0, 0, 0, 1, 1],
    "akn.176.016.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.176.125.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.176.216.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.178.079.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.179.031.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.179.087.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.179.092.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.179.178.left.avi": [0, 0, 0, 1, 0, 1],
    "akn.182.020.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.182.029.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.182.050.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.182.130.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.184.139.left.avi": [0, 0, 0, 1, 0, 1],
    "akn.185.061.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.185.084.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.186.020.left.avi": [0, 0, 0, 1, 0, 0],
    "akn.186.105.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.186.156.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.187.023.left.avi": [0, 0, 0, 1, 0, 0],
    "akn.188.262.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.189.239.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.189.241.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.190.381.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.190.410.left.avi": [0, 0, 0, 1, 0, 0],
    "akn.193.073.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.197.093.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.198.136.left.avi": [0, 0, 0, 1, 0, 0],
    "akn.199.020.left.avi": [0, 0, 0, 1, 0, 1],
    "akn.200.010.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.204.006.left.avi": [0, 0, 0, 1, 0, 1],
    "akn.211.002.left.avi": [0, 0, 0, 1, 0, 1],
    "akn.211.006.left.avi": [0, 0, 0, 1, 0, 1],
    "akn.212.018.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.212.100.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.212.134.left.avi": [1, 0, 0, 0, 0, 1],
    "akn.212.179.left.avi": [0, 0, 0, 1, 0, 1],
    "akn.212.221.left.avi": [0, 0, 0, 1, 0, 1],
    "akn.212.237.left.avi": [0, 0, 0, 1, 0, 1],
    "akn.215.153.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.215.175.left.avi": [0, 0, 0, 1, 0, 1],
    "akn.216.133.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.217.019.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.217.104.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.219.014.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.219.149.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.219.183.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.220.011.left.avi": [0, 0, 0, 1, 0, 0],
    "akn.220.030.left.avi": [0, 0, 0, 1, 0, 0],
    "akn.220.132.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.221.004.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.222.007.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.222.029.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.223.098.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.224.017.left.avi": [0, 0, 0, 1, 0, 0],
    "akn.224.115.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.228.171.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.228.225.left.avi": [0, 0, 0, 1, 0, 1],
    "akn.228.244.left.avi": [0, 0, 0, 1, 0, 1],
    "akn.229.188.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.232.051.left.avi": [0, 0, 1, 0, 0, 0],
    "akn.232.074.left.avi": [0, 0, 0, 1, 0, 0],
    "akn.232.162.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.233.113.left.avi": [0, 0, 1, 0, 0, 0],
    "akn.233.148.left.avi": [0, 0, 1, 0, 0, 0],
    "akn.233.152.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.233.182.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.240.027.left.avi": [0, 0, 1, 0, 0, 0],
    "akn.240.095.left.avi": [0, 0, 1, 0, 0, 0],
    "akn.240.184.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.243.043.left.avi": [0, 0, 0, 1, 0, 1],
    "akn.243.177.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.243.206.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.243.207.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.243.211.left.avi": [0, 0, 0, 1, 0, 0],
    "akn.244.049.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.244.141.left.avi": [1, 0, 0, 0, 0, 1],
    "akn.244.165.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.245.022.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.247.011.left.avi": [0, 0, 0, 1, 0, 1],
    "akn.250.022.left.avi": [0, 0, 0, 1, 0, 0],
    "akn.250.140.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.250.300.left.avi": [0, 0, 0, 1, 0, 1],
    "akn.251.029.left.avi": [0, 0, 0, 1, 0, 0],
    "akn.256.049.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.261.088.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.263.136.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.266.024.left.avi": [0, 1, 0, 0, 0, 1],
    "akn.266.396.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.266.430.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.267.004.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.267.064.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.267.135.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.267.139.left.avi": [0, 0, 0, 1, 0, 0],
    "akn.269.036.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.269.085.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.269.094.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.269.111.left.avi": [0, 0, 0, 1, 0, 1],
    "akn.270.197.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.271.109.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.271.132.left.avi": [0, 0, 0, 1, 0, 0],
    "akn.272.018.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.272.139.left.avi": [0, 0, 0, 1, 0, 0],
    "akn.273.009.left.avi": [0, 0, 0, 1, 0, 0],
    "akn.273.014.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.273.056.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.273.074.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.275.015.left.avi": [0, 0, 0, 1, 0, 1],
    "akn.275.115.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.279.026.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.280.006.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.281.024.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.281.131.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.282.083.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.283.065.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.283.175.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.283.209.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.289.008.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.289.041.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.289.048.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.289.057.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.289.058.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.289.069.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.292.005.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.292.154.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.294.156.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.295.027.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.295.194.left.avi": [0, 0, 0, 1, 0, 0],
    "akn.308.081.left.avi": [0, 0, 1, 0, 0, 0],
    "akn.308.104.left.avi": [0, 0, 1, 0, 0, 0],
    "akn.308.139.left.avi": [0, 0, 1, 0, 0, 0],
    "akn.343.073.left.avi": [0, 1, 0, 0, 1, 0],
    "akn.343.399.left.avi": [0, 1, 0, 0, 1, 0],
    "akn.344.034.left.avi": [0, 0, 1, 0, 1, 0],
    "akn.344.050.left.avi": [0, 0, 1, 0, 1, 0],
    "akn.344.106.left.avi": [0, 0, 1, 0, 1, 0],
    "akn.345.006.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.345.202.left.avi": [0, 0, 1, 0, 0, 0],
    "akn.389.266.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.389.282.left.avi": [0, 0, 1, 0, 0, 0],
    "akn.391.181.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.391.241.left.avi": [0, 0, 1, 0, 0, 0],
    "akn.393.054.left.avi": [0, 1, 0, 0, 1, 1],
    "akn.393.224.left.avi": [0, 0, 1, 0, 1, 0],
    "akn.393.331.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.394.008.left.avi": [0, 0, 1, 0, 0, 0],
    "akn.394.018.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.394.123.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.395.047.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.414.251.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.419.002.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.420.236.left.avi": [0, 0, 1, 0, 0, 0],
    "akn.421.209.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.421.241.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.421.252.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.421.255.left.avi": [0, 0, 1, 0, 0, 0],
    "akn.421.412.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.425.073.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.429.105.left.avi": [0, 0, 1, 0, 0, 0],
    "akn.430.500.left.avi": [0, 1, 0, 0, 0, 0],
    "akn.431.139.left.avi": [0, 0, 1, 0, 0, 0],
}

def process_frame(frame):
    # resize the frame to fit my screen better
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    # frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # now we can see the light
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame, (0, 0, 130), (255, 200, 255))

    # generate mask for white. shoulder the burden on canny
    mask = cv2.inRange(frame, (0, 0, 130), (255, 200, 255))
    frame = cv2.bitwise_and(frame, frame, mask=mask)

    # get the edges and find contours
    h, s, v = cv2.split(frame)
    frame = cv2.Canny(v, 100, 200)
    frame, contours, hierarchy = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    frame = cv2.drawContours(frame, contours, -1, (0, 0, 255))

    shitlist = []
    for contour in contours:
        epsilon = .02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        (x, y, w, h) = cv2.boundingRect(approx)
        if w < h:
            continue

        if x < frame.shape[1] * 2 / 3 and x > frame.shape[1] * 1 / 3:
            continue

        if float(cv2.contourArea(approx)) < 400:
            continue

        # do extent
        extent = float(cv2.contourArea(approx)) / (w * h)
        if extent < 0.5:
            continue

        frame = cv2.drawContours(frame, [approx], -1, (255, 0, 255))
        drawRectangle(frame, x, y, w, h)
        drawRectangle(frame, x - 10, y - 10, w + 20, h + 20, (0, 255, 0))
        shitlist.append((x, y, w, h))
        cv2.imshow('out', frame)
        cv2.waitKey(0)

    return shitlist

def render(frame, center_x, center_y):
    squareList = process_frame(frame)
    if squareList:
        for item in squareList:
            x, y, w, h = item

            img1 = frame[x:x+h, y:y+w]
            img2 = cv2.imread('speedbump.png')

            sift = cv2.ORB_create()

            kp1, des1 = sift.detectAndCompute(img1,None)
            kp2, des2 = sift.detectAndCompute(img2,None)

            # create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Match descriptors.
            matches = bf.match(des1,des2)

            # Sort them in the order of their distance.
            matches = sorted(matches, key = lambda x:x.distance)

            # Draw first 10 matches.
            img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], None, flags=2)
            cv2.imshow('out', img1)
            # plt.imshow(img3),plt.show()

            # Draw focal point
            cv2.waitKey(1)

    return False

def file_get_contents(filename):
    with open(filename) as f:
        return f.read()

def process_video(file_name):
    index = 0
    text_file = "{:s}.txt".format(file_name[:-9])
    center_x, center_y = file_get_contents(text_file).split(" ")
    center_x = int(center_x)
    center_y = int(center_y)

    cap = cv2.VideoCapture(file_name)
    print("Reading file {:s}".format(file_name))
    while(cap.isOpened()):
        ret, frame = cap.read()
        index += 1
        if not ret:
            break

        if index != 105: continue
        print("FRAME NUMBER {:d}".format(index))
        if render(frame, center_x, center_y):
            print("MATCH ON FRAME {:d}".format(index))
            return (file_name, True)
    return (file_name, False)

files = []

for fileObj in os.listdir("trainset"):
    if fileObj.endswith(".avi"):
        files.append(os.path.join("trainset", fileObj))

files = ["trainset/akn.031.029.left.avi"]
for file_name in files:
    key = os.path.basename(file_name)
    if data_info[key][3] != 1: continue
    file_name, passes = process_video(file_name)

    if passes:
        if data_info[key][3] == 1:
            print("DETECTED")
        else:
            print("FALSE POSITIVE")
    elif data_info[key][3] == 1:
        print("MISSED DETECTION")
    else:
        print("NOTHING TO DETECT")

cv2.destroyAllWindows()
