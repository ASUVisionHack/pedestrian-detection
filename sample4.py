import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
import os

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

    def isValid(m, l):

        if l[1] < center_y or l[3] < center_y:
            return False

        height = abs(l[3] - l[1])
        width = abs(l[2] - l[0])
        dx = (center_x - l[0])
        dy = abs(center_y - l[1])
        slope =  dx / dy

        if not dy > 0:
            return False

        if height < 20 or height > 50 or width > 100:
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

    if len(filtered_lines) > 7:
        # print("We're passing")
        ys.sort(key=int)
        # print(ys)
        y_delta = abs(ys[0] - ys[-1])
        # print(y_delta)
        if(y_delta < 70):
            return True

    # print("Number of lines: {:d}".format(len(filtered_lines)))
    # drawn_img = lsd.drawSegments(gray,np.array(filtered_lines))

    # Draw focal point
    # cv2.circle(drawn_img,(int(center_x),int(center_y)), 20, (0,0,255), -1)
    # cv2.imshow("Image", drawn_img)
    # cv2.waitKey(0)


def file_get_contents(filename):
    with open(filename) as f:
        return f.read()

files = []
for file in os.listdir("zebra"):
    if file.endswith(".avi"):
        files.append(os.path.join("zebra", file))

# files = ["zebra/akn.088.141.left.avi"]
index = 0
for file_name in files:
    text_file = "{:s}.txt".format(file_name[:-9])
    center_x, center_y = file_get_contents(text_file).split(" ")
    center_x = int(center_x)
    center_y = int(center_y)

    cap = cv2.VideoCapture(file_name)
    print("Reading file {:s}".format(file_name))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        index += 1
        # if index != 24: continue
        # if index != 628: continue
        # print("Index Number: {:d}".format(index))
        if render(frame, center_x, center_y):
            print("TRUE")
            break
    cap.release()

cv2.destroyAllWindows()
