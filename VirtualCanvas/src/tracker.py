import utils
import numpy as np
import cv2 as cv

MIN_OBJECT_AREA = 180
RED = (0, 0, 255)  # BGR format

if __name__ == "__main__":
    pointerCol = utils.loadPointerCol()  # loading the range of pen col

    cap = utils.getCameraFeed()  # get webcam feed
    window_name = "Tracker"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)  # creating window

    while cap.isOpened():
        _, frame = cap.read()
        # mirror flipping the frame
        frame = cv.flip(frame, 1)

        # converting the image into hsv
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        lowerRange = pointerCol[0]
        upperRange = pointerCol[1]

        mask = cv.inRange(hsv, lowerRange, upperRange)
        # nose reduction
        mask = utils.noiseReduction(mask)

        # get the outline of the object
        contours, _ = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if contours and cv.contourArea(max(contours, key=cv.contourArea)) > MIN_OBJECT_AREA:
            c = max(contours, key=cv.contourArea)
            # get the dimension for recangle binding around the object
            x, y, w, h = cv.boundingRect(c)
            # add the rectangle
            cv.rectangle(frame, (x, y), (x+w, y+h), RED, 2)

        # add modified image to the window
        cv.imshow(window_name, frame)

        k = cv.waitKey(1)
        if k == 27:
            break
    # close the window
    utils.endWindow(cap)
