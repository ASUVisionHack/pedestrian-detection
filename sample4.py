import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt

def render(image):
    center_x = 946 * .6
    center_y = 540 * .6

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (0, 0), fx=0.6, fy=0.6)
    # gray = gray[250:500, 0:1920]

    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(gray)[0]
    filtered_lines = []

    xs = []
    ys = []

    def isCentered(m, l):

        if l[1] < center_y or l[3] < center_y:
            return False

        dx = (center_x - l[0])
        dy = abs(center_y - l[1])
        slope =  dx/dy

        # print("dx: {:f}, dy: {:f}".format(dx, dy))
        # print("real slope: {:f}, expected slope: {:f}".format(m, slope))
        # cv2.line(gray, (int(l[0]),int(l[1])), (int(center_x),int(center_y)), (255,0,0), 1)

        # Check result vs expected is within 10%
        return (abs(1 - m/slope) * 100) < 10

    for line in lines:
        l = line[0]

        dx = l[0] - l[2] if l[1] < l[3] else l[2] - l[0]
        dy = abs(l[3] - l[1])

        if dy > 20:
            slope = dx / dy
            print("x: {:f},  y: {:f}, slope: {:f}".format(dx, dy, slope))
            filtered_lines.append(line)
            # if isCentered(slope, l):
                # filtered_lines.append(line)

        # if y_d < x_d and y_d < 10 and x_d > 20 and x_d < 60:
        #     print("{:f} - {:f}".format(x_d, y_d))
        #     filtered_lines.append(line)

    drawn_img = lsd.drawSegments(gray,np.array(filtered_lines))
    # cv2.circle(drawn_img,(int(center_x),int(center_y)), 20, (0,0,255), -1)

    cv2.imshow("Image", drawn_img)
    cv2.waitKey(0)

cap = cv2.VideoCapture('zebra/akn.031.037.left.avi')
while(cap.isOpened()):
    ret, frame = cap.read()
    render(frame)
cap.release()

# image = cv2.imread('sample-img-1.png')
# render(image)
cv2.destroyAllWindows()
