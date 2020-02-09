import cv2
import numpy as np

def nothing(x):
    pass

# import picture
img = cv2.imread('abstractLines.bmp')
# create trackbar
cv2.namedWindow('threshold', cv2.WINDOW_FREERATIO)
cv2.createTrackbar('red', 'threshold', 0, 127, nothing)
cv2.createTrackbar('green', 'threshold', 0, 255, nothing)
cv2.createTrackbar('blue', 'threshold', 0, 10, nothing)
#switch = '0:OFF \n 1:ON'
#cv2.createTrackbar(switch, 'threshold', 0, 1, nothing)

while True:
    # convert colored picture to grayscale
    imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    redThresh = cv2.getTrackbarPos('red', 'threshold')
    greenThresh = cv2.getTrackbarPos('green', 'threshold')
    blueThresh = cv2.getTrackbarPos('blue', 'threshold')
    #threshSwitch = cv2.getTrackbarPos(switch, 'threshold')
    # thresholding the image accordingly sety by RGB values?
    ret, thresh = cv2.threshold(imGray, redThresh, greenThresh, blueThresh)
    # if contours are to be found using grayscale image directly... Unsupported format or combination of formats
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawcnt = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    cv2.imshow('threshold', drawcnt)
    key = cv2.waitKey(1) & 0xFF
    if key==27:
        break

cv2.destroyAllWindows()
