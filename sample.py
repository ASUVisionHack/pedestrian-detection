import numpy as np
import cv2
import imutils

def render(ret, frame):
    frame = imutils.resize(frame, height = 500)

    frame = cv2.blur(frame,(4,4))
    height, width, channels = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    cv2.imshow("Happy Days", gray)
    # cv2.waitKey(0)

    edged = cv2.Canny(gray, 8, 50)

    ret, edged = cv2.threshold(edged, 127, 255, 0)
    image, cnts, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        # approximate the contour
        epsilon = .04 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        print('{:f}, {:f}'.format(epsilon, len(approx)))

        if len(approx) > 3:
            cv2.drawContours(image, [approx], -1, (255, 255, 0), cv2.LINE_4, 8)

    cv2.imshow("Happy Days", image)
    cv2.waitKey(0)

# cap = cv2.VideoCapture('akn.393.054.left.avi')

# while(cap.isOpened()):
    # ret, frame = cap.read()
frame = cv2.imread('sample-img-1.png')
render(None, frame)

# cap.release()
cv2.destroyAllWindows()
