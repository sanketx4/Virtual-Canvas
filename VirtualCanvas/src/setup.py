import utils
import time
import cv2 as cv
import numpy as np
import os

# image ratio factor
fractionX, fractionY = 0.5, 0.5

if __name__ == "__main__":
    cap = utils.getCameraFeed()  # start the webcam
    utils.wait()  # wait for 2 sec to start the webcam

    utils.createColorSelctionWindow()  # create the setup GUI window

    print("Press S to save the pen values")

    # Loop of video capture
    while cap.isOpened():
        ret, frame = cap.read()  # reading from camera feed
        if not ret:
            break

        # fliping the image horizontally because the camera feed takes mirror image as input
        frame = cv.flip(frame, 1)

        # get HSV format image from BGR
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # get the threshold color saturation value
        lowerRange, upperRange = utils.getColorRange()
        # Mask the image for lower and upper range
        mask = cv.inRange(hsv, lowerRange, upperRange)
        # black and white masked image
        res = cv.bitwise_and(frame, frame, mask=mask)
        # convert the grayscale image to colorful image
        bgrMask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

        # create a row of different frames
        stacked = np.hstack((bgrMask, frame, res))

        # show the updated image in the window
        cv.imshow(utils.window_name, cv.resize(
            stacked, None, fx=fractionX, fy=fractionY))

        k = cv.waitKey(1)
        if k == 27:
            # if ESC pressed then exit
            break

        if k == ord('s'):
            # if s pressed then save the lower and uper range value of the color
            try:
                os.mkdir("data")
            except:
                # Directory already exist
                pass
            np.save('data/pointerCol', np.array([lowerRange, upperRange]))
            break

    # close the windows
    utils.endWindow(cap)
