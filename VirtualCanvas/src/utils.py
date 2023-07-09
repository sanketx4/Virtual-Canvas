import cv2 as cv
import numpy as np
import time


window_name = "Setup Pen Color"


# create the sliding window for setup
def createColorSelctionWindow():
    def foo(x):
        pass

    cv.namedWindow(window_name)
    cv.createTrackbar("L-H", window_name, 0, 179, foo)
    cv.createTrackbar("L-S", window_name, 0, 255, foo)
    cv.createTrackbar("L-V", window_name, 0, 255, foo)
    cv.createTrackbar("U-H", window_name, 179, 179, foo)
    cv.createTrackbar("U-S", window_name, 255, 255, foo)
    cv.createTrackbar("U-V", window_name, 255, 255, foo)


# Extract the lower and upper hsv value according to the slider position
def getColorRange():
    l_h = cv.getTrackbarPos("L-H", window_name)
    l_s = cv.getTrackbarPos("L-S", window_name)
    l_v = cv.getTrackbarPos("L-V", window_name)
    u_h = cv.getTrackbarPos("U-H", window_name)
    u_s = cv.getTrackbarPos("U-S", window_name)
    u_v = cv.getTrackbarPos("U-V", window_name)
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])
    return lower_range, upper_range


# capture feed from webcam
def getCameraFeed():
    WEB_CAM = 0
    return cv.VideoCapture(WEB_CAM)


# idealy wait for given time (default = 2 sec)
def wait(waitTime=2):
    time.sleep(waitTime)


# stop the webcam feed and closes all the open window
def endWindow(cap):
    cap.release()
    cv.destroyAllWindows()


# looks for pointer col in data folder
def loadPointerCol():
    try:
        pointerCol = np.load('data/pointerCol.npy')
        return pointerCol
    except:
        raise ValueError("Pointer Color range Not found! Do setup first")
        exit()


# standerd morphing techniques in computer vision for noise reduction
def noiseReduction(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.erode(mask, kernel, iterations=1)
    mask = cv.dilate(mask, kernel, iterations=2)
    return mask
