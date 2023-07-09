import utils
import numpy as np
import cv2 as cv
import enum

MIN_OBJECT_AREA = 180
RED = (0, 0, 255)  # BGR format
BLUE = (255, 0, 0)  # BGR format
GREEN = (0, 255, 0)  # BGR format

if __name__ == "__main__":
    pointerCol = utils.loadPointerCol()  # loading the range of pen col

    cap = utils.getCameraFeed()  # get webcam feed
    window_name = "Tracker"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)  # creating window

    x_coordinate, y_coordinate = 0, 0  # initial coordinates of object
    canvas = None  # the board over which we are going to write
    currenColor = RED
    thickNess = 5
    isPointer = True

    while cap.isOpened():
        _, frame = cap.read()
        k = cv.waitKey(1)
        # mirror flipping the frame
        frame = cv.flip(frame, 1)

        # converting the image into hsv
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # if no canvas then initialize a frame size zero canvas
        if canvas is None:
            canvas = np.zeros_like(frame)

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

            # update pointer coordinate in canvas
            if x_coordinate == 0 and y_coordinate == 0:
                x_coordinate, y_coordinate = x, y
            else:
                # join line from previous iterations position to current position
                if k == ord('r'):
                    currenColor = RED
                    thickNess = 5
                elif k == ord('b'):
                    currenColor = BLUE
                    thickNess = 5
                elif k == ord('g'):
                    currenColor = GREEN
                    thickNess = 5
                elif k == ord('e'):
                    currenColor = [0, 0, 0]
                    thickNess = 20
                if not isPointer:
                    canvas = cv.line(
                        canvas, (x_coordinate, y_coordinate), (x, y), currenColor, thickNess)
            x_coordinate, y_coordinate = x, y
        else:
            x_coordinate, y_coordinate = 0, 0  # if unable to detect pointer then reset

        _, mask = cv.threshold(cv.cvtColor(
            canvas, cv.COLOR_BGR2GRAY), 20, 255, cv.THRESH_BINARY)
        fg = cv.bitwise_and(canvas, canvas, mask=mask)
        bg = cv.bitwise_and(frame, frame, mask=cv.bitwise_not(mask))
        frame = cv.add(fg, bg)

        # add modified image to the window
        cv.imshow(window_name, frame)

        if k == ord('c'):
            canvas = None
        if k == 27:
            break
        if k == ord('i'):
            isPointer = not isPointer
    # close the window
    utils.endWindow(cap)
