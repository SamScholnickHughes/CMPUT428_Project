import cv2
import numpy as np

class Tracker():
    def __init__(self, img, corners):
        # initialize your tracker with the first frame from the sequence and
        # the corresponding corners from the ground truth
        # this function does not return anything
        self.old_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.p0 = corners.T.astype(np.float32)


    def updateTracker(self, img):
        # update your tracker with the current image and return the current corners
        # at present it simply returns the actual corners with an offset so that
        # a valid value is returned for the code to run without errors
        # this is only for demonstration purpose and your code must NOT use actual corners in any way
        frame_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # frame_img = img
        # parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (32,32),
                    maxLevel = 8,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 9, 0.02555))
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, frame_img, self.p0, None, **lk_params)
        self.old_frame = frame_img.copy()
        self.p0 = p1.copy()

        return p1.T
    
    def drawRegion(self, img, corners, color, thickness=1):
        # draw the bounding box specified by the given corners
        for i in range(4):
            p1 = (int(corners[0, i]), int(corners[1, i]))
            p2 = (int(corners[0, (i + 1) % 4]), int(corners[1, (i + 1) % 4]))
            cv2.line(img, p1, p2, color, thickness)