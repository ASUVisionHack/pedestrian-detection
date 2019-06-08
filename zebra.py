import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
import os
import math

data_info = {
    "akn.186.054.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.160.227.left.avi": [1, 0, 0, 0, 0, 1],
    "akn.220.010.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.394.163.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.279.235.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.391.185.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.220.031.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.289.063.left.avi": [0, 0, 0, 0, 1, 1],
    "akn.160.230.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.240.107.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.215.174.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.184.110.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.393.269.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.276.240.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.289.109.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.174.119.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.179.103.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.213.002.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.151.155.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.187.029.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.394.005.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.295.193.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.289.055.left.avi": [0, 0, 0, 0, 1, 1],
    "akn.282.018.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.176.148.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.081.021.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.345.215.left.avi": [0, 0, 0, 0, 1, 1],
    "akn.136.213.left.avi": [1, 0, 0, 0, 0, 1],
    "akn.226.130.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.215.131.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.112.004.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.278.125.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.183.042.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.249.032.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.182.004.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.292.002.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.160.030.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.031.041.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.215.159.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.344.046.left.avi": [0, 0, 0, 0, 1, 1],
    "akn.160.215.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.261.008.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.178.218.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.266.392.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.268.082.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.195.097.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.394.115.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.084.020.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.391.411.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.212.238.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.245.047.left.avi": [1, 0, 0, 0, 0, 1],
    "akn.393.243.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.090.121.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.190.394.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.184.109.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.289.047.left.avi": [0, 0, 0, 0, 1, 1],
    "akn.256.137.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.267.150.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.344.029.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.345.037.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.393.029.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.244.272.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.276.246.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.165.068.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.139.044.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.192.107.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.393.202.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.223.214.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.246.061.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.182.062.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.185.070.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.189.245.left.avi": [1, 0, 0, 0, 0, 1],
    "akn.269.097.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.081.054.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.394.210.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.217.155.left.avi": [1, 0, 0, 0, 0, 1],
    "akn.241.131.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.233.260.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.079.064.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.394.216.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.343.101.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.145.141.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.166.049.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.186.063.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.269.115.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.223.010.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.283.210.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.113.142.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.174.093.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.276.214.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.174.097.left.avi": [0, 0, 0, 0, 1, 1],
    "akn.243.231.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.275.013.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.240.123.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.241.130.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.389.270.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.160.370.left.avi": [1, 0, 0, 0, 0, 1],
    "akn.344.090.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.224.016.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.176.054.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.145.038.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.078.047.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.211.011.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.136.038.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.275.072.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.174.094.left.avi": [0, 0, 0, 0, 1, 1],
    "akn.344.074.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.391.422.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.263.023.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.279.014.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.031.008.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.430.048.left.avi": [0, 0, 0, 0, 1, 0],
    "akn.273.044.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.196.065.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.421.450.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.224.040.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.187.170.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.398.102.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.182.138.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.223.015.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.214.036.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.419.006.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.232.050.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.256.117.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.241.109.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.241.044.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.160.015.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.136.274.left.avi": [1, 0, 0, 0, 0, 1],
    "akn.243.162.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.275.080.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.226.001.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.136.059.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.176.171.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.184.071.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.213.022.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.279.044.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.289.068.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.390.126.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.226.080.left.avi": [1, 0, 0, 0, 1, 0],
    "akn.389.020.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.222.004.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.346.244.left.avi": [1, 0, 0, 0, 1, 1],
    "akn.393.334.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.152.144.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.174.103.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.228.125.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.419.022.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.250.265.left.avi": [1, 0, 0, 0, 0, 1],
    "akn.268.054.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.273.086.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.219.035.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.192.122.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.223.057.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.207.058.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.223.019.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.289.046.left.avi": [0, 0, 0, 0, 1, 1],
    "akn.425.139.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.199.016.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.229.170.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.250.230.left.avi": [1, 0, 0, 0, 0, 1],
    "akn.174.149.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.215.162.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.160.379.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.211.008.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.275.009.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.391.417.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.394.111.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.131.019.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.182.108.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.229.086.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.182.002.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.185.062.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.220.009.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.132.075.left.avi": [1, 0, 0, 0, 0, 1],
    "akn.282.069.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.184.002.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.189.249.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.289.007.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.178.106.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.174.089.left.avi": [0, 0, 0, 0, 1, 1],
    "akn.279.027.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.345.074.left.avi": [0, 0, 0, 0, 1, 1],
    "akn.224.030.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.174.140.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.279.020.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.048.067.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.232.047.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.082.001.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.246.009.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.186.178.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.184.150.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.279.032.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.269.082.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.277.119.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.213.052.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.186.064.left.avi": [0, 0, 0, 0, 0, 1],
    "akn.308.095.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.185.171.left.avi": [1, 0, 0, 0, 0, 0],
    "akn.233.017.left.avi": [0, 0, 0, 0, 0, 0],
    "akn.213.357.left.avi": [1, 0, 0, 0, 0, 0],
}

def render(image, center_x, center_y):
    scaler = .6
    center_x = center_x * scaler
    center_y = center_y * scaler

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (0, 0), fx=scaler, fy=scaler)

    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(gray)[0]
    filtered_lines = []

    ys = []
    xs = []

    def isValid(m, l):

        if l[1] < center_y or l[3] < center_y:
            return False

        if max(l[1], l[3]) > 570:
            return False

        height = abs(l[3] - l[1])
        width = abs(l[2] - l[0])
        dx = (center_x - l[0])
        dy = abs(center_y - l[1])

        if not dy > 0:
            return False

        slope =  dx / dy
        if height < 22 or height > 50 or width > 100:
            return False

        diff = abs(m - slope)
        # print("dx: {:f}. dy: {:f}, actual: {:f}, derived: {:f}, diff: {:f}".format(height, width, m, slope, diff))

        return diff < .35

    # print("---------------------------------------------------")
    for line in lines:
        l = line[0]

        dx = l[0] - l[2] if l[1] < l[3] else l[2] - l[0]
        dy = abs(l[3] - l[1])

        if dy > 0:
            slope = dx / dy
            if isValid(slope, l):
                # print("dx: {:f}. dy: {:f}, slope: {:f}".format(dx, dy, slope))
                filtered_lines.append(line)
                ys.append(min(l[1], l[3]))
                xs.append(min(l[0], l[2]))

    drawn_img = lsd.drawSegments(gray,np.array(filtered_lines))
    cv2.imshow("Image", drawn_img)
    cv2.waitKey(0)
    if len(filtered_lines) > 7:
        ys.sort(key=int)
        xs.sort(key=int)
        # print(ys)
        # print(xs)
        y_delta = abs(ys[0] - ys[-1])

        #Calculate Largest Distance Between points
        xs = [t - s for s, t in zip(xs, xs[1:])]
        xs.sort(key=int)
        # print(xs)
        if xs[-1] > 290:
            return False

        # print(y_delta)
        if(y_delta < 60):
            af_line_list = []
            for af_line in filtered_lines:
                af_line = af_line[0]

                d2 = math.pow((center_x - (af_line[0] + af_line[2])/2), 2) + math.pow((center_y - (af_line[1] + af_line[3])/2), 2)
                af_line_list.append(d2)

            # print(np.var(np.array(af_line_list)))
            if np.var(np.array(af_line_list)) > 300000000:
                # print("Bailing due to Daniel Math")
                return False

            # print("Number of lines: {:d}".format(len(filtered_lines)))

            # Draw focal point
            # cv2.circle(drawn_img,(int(center_x),int(center_y)), 20, (0,0,255), -1)

            # return True


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
    # print("Reading file {:s}".format(file_name))
    while(cap.isOpened()):
        ret, frame = cap.read()
        index += 1
        if not ret:
            break

        if render(frame, center_x, center_y):
            # print("MATCH ON FRAME {:d}".format(index))
            return (file_name, True)
    return (file_name, False)

files = []

for fileObj in os.listdir("validationset"):
    if fileObj.endswith(".avi"):
        files.append(os.path.join("validationset", fileObj))

# files = ["validationset/akn.240.107.left.avi"]
for file_name in files:
    key = os.path.basename(file_name)
    if data_info[key][5] == 1:
        file_name, passes = process_video(file_name)
        # if passes:
        #     if data_info[key][5] == 1:
            # print("ALL GOOD")
        #     else:
        #         print("FALSE POSITIVE")
        # else:
            # print("MISSED DETECTION")

cv2.destroyAllWindows()
