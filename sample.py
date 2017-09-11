import numpy as np
import cv2
import imutils

def render(gray):
    gray = imutils.resize(gray, height = 400)

    gray = cv2.blur(gray,(4,4))
    height, width, channels = frame.shape
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 10, 20, 10)

    # cv2.imshow("Happy Days", gray)
    # cv2.waitKey(0)

    edged = cv2.Canny(gray, 8, 50)

    ret, edged = cv2.threshold(edged, 127, 255, 0)
    image, cnts, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    matches = []
    for c in cnts:
        # approximate the contour
        epsilon = .02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        count = len(approx)

        if ar < 1.5 and w > 25 and h > 30 and h < 60 and y > 200 and count > 3 and x > 200 and x < 500:
            # print('epsilon: {:f}, num: {:f}'.format(epsilon, count))
            # print('x: {:f}, y: {:f}, w: {:f}, h: {:f}, ar: {:f}'.format(x, y, w, h, ar))

            cv2.drawContours(image, [approx], -1, (255, 255, 0), cv2.LINE_4, 8)
            cv2.imshow("Happy Days", image)

            if(len(matches) == 0):
                matches.append({
                    'approxs' : [approx],
                    'y': y,
                })
            else:
                match = False
                for v in matches:
                    if abs(v['y'] - y) < 10:
                        v['approxs'].append(approx)
                        match = True
                if match is False:
                    matches.append({
                        'approxs' : [approx],
                        'y': y,
                    })

    passes = False
    for v in matches:
        if passes is False and len(v['approxs']) > 3:
            passes = True

    if passes:
        print("We have Crosswalk!")
    else:
        print("No Crosswalk")
    cv2.imshow("Happy Days", image)
    cv2.waitKey(0)

cap = cv2.VideoCapture('zebra/akn.088.141.left.avi')

while(cap.isOpened()):
    ret, frame = cap.read()
    render(frame)
cap.release()

# frame = cv2.imread('sample-img-1.png')
# render(frame)
#
# frame = cv2.imread('sample-img-2.png')
# render(frame)
cv2.destroyAllWindows()
