#################################################################################################################################
### This file implements color specific sphere tracking. It can potentially be used for tracking recovery by object detection ###
#################################################################################################################################

#Requires refactoring. This is just a "proof of concept" implementation

import cv2
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
import LKTracker
from math import sqrt

#####HSV Colour Ranges#################
#If the ball is red (0-10) or (170-180)
redLowMask = (0,130,80)
redHighMask = (20, 255, 255)

#If the ball is blue
# blueLowMask = (100, 127, 67)
# blueHighMask = (179, 188, 90)
blueLowMask = (90, 200, 0)
blueHighMask = (180, 255, 200)

#If the ball is orange
yellowLowMask = (20, 60, 195)
yellowHighMask = (255, 255, 255)

#If the ball is green
greenLowMask= (30, 145, 0)
greenHighMask= (90, 255, 255)
########################################
VERBOSE = True

class Tracker:

    def __init__(self, col, num_pts):
        self.points = []
        for i in range(num_pts):
            self.points.append([0, 0, 0])
        thread = threading.Thread(target=self.TrackerThread, args=([col]), daemon=True)
        thread.start()

    def TrackerThread(self, col):
        print(col)
        print("Tracker Started")
        # Get the camera
        vc = cv2.VideoCapture(1)
        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False
        while rval:
            # Handle current frame
            rval, frame = vc.read()
            circles, frame = self.GetLocation(frame, col)
            print(circles)
            if VERBOSE: self.DrawCircles(frame, circles, (255, 255, 0))

            if circles is not None and len(circles) >= len(self.points):
                for i in range(len(self.points)):
                    self.points[i] = circles[i]

                # Shows the original image with the detected circles drawn.
            if VERBOSE: cv2.imshow("Result", frame)

            # check if esc key pressed
            key = cv2.waitKey(20)
            if key == 27:
                break
        
        vc.release()
        cv2.destroyAllWindows()
        print("Tracker Ended")

    def GetLocation(self, frame, color):
        print(color)
        # Uncomment for gaussian blur
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        # blurred = cv2.medianBlur(frame,11)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if color == 'r':
            # Red Tracking
            mask = cv2.inRange(hsv, redLowMask, redHighMask)
        if color == 'y':
            # Orange Tracking
            mask = cv2.inRange(hsv, yellowLowMask, yellowHighMask)
        if color == 'b':
            # Blue Tracking
            mask = cv2.inRange(hsv, blueLowMask, blueHighMask)
        if color == 'g':
            # Green Tracking
            mask = cv2.inRange(hsv, greenLowMask, greenHighMask)
        # Perform erosion and dilation in the image (in 11x11 pixels squares) in order to reduce the "blips" on the mask
        # mask = cv2.erode(mask, np.ones((11, 11),np.uint8), iterations=2)
        # mask = cv2.dilate(mask, np.ones((11, 11),np.uint8), iterations=5)
        # Mask the blurred image so that we only consider the areas with the desired colour
        masked_blurred = cv2.bitwise_and(blurred,blurred, mask= mask)
        # masked_blurred = cv2.bitwise_and(frame,frame, mask= mask)
        # Convert the masked image to gray scale (Required by HoughCircles routine)
        result = cv2.cvtColor(masked_blurred, cv2.COLOR_BGR2GRAY)
        # Detect circles in the image using Canny edge and Hough transform
        circles = cv2.HoughCircles(result, cv2.HOUGH_GRADIENT, 1.5, 300, param1=30, param2=15, minRadius=5, maxRadius=300)
        return circles, masked_blurred
            
    def DrawCircles(self, frame, circles, dotColor):
        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                #print("Circle: " + "("+str(x)+","+str(y)+")")
                # draw the circle in the output image, then draw a rectangle corresponding to the center of the circle
                # The circles and rectangles are drawn on the original image.
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), dotColor, -1)

def _get_bbox(pt):
    size = 15
    bbox = [[pt[0]-size, pt[1]-size], 
            [pt[0]+size, pt[1]-size],
            [pt[0]+size, pt[1]+size],
            [pt[0]-size, pt[1]+size]]
    # print(bbox, pt)
    return np.array(bbox).T

def _get_center(pts):
    pts = np.concatenate((pts.T, np.ones([4, 1])), axis=1) 

    l1 = np.cross(pts[0], pts[2])
    l2 = np.cross(pts[1], pts[3])

    pm = np.cross(l1, l2)

    norm_pm = pm / pm[2]
    norm_pm = norm_pm.astype(np.int32)

    return norm_pm[:2]
            
if __name__ == "__main__":
    # VERBOSE = True
    # print("Tracker Setup")
    # # tracker = Tracker(['r', 'b', 'g', 'y'])
    # tracker = Tracker('r', 4)
    # print("Moving on")
    # while True:
    #     print("Point is at: "+str(tracker.points))
    #     time.sleep(2)
    num_pts = 4
    show = True
    cam = cv2.VideoCapture(0)

    ret, frame = cam.read()
    # print(frame.shape)

    fig, ax = plt.subplots() 
    ax.imshow(frame) 
    ax.axis('off') 
        
    plt.title("Image") 
    
    pts = plt.ginput(num_pts)
    plt.close()

    trackers = []
    for i in range(num_pts):
        trackers.append(LKTracker.Tracker(frame, _get_bbox(pts[i])))


    print(frame.shape)
    while True:
        ret, frame = cam.read()
        try:
            cframe = frame.copy()
        except AttributeError:
            cv2.destroyAllWindows()
            break
        if not ret:
            print("failed to grab frame")
            break

        # Applying the Canny Edge filter
        # frame = cv2.Canny(frame, t_lower, t_upper)
        frame = cv2.medianBlur(frame,5)
        corners = np.zeros([2, num_pts])
        for i in range(num_pts):
            tracker = trackers[i]
            c = tracker.updateTracker(frame)
            corners[:, i] = _get_center(c)
            # print(_get_center(c))
        # print(corners)
        # raise Exception
        
        draw_corners = np.array([corners[:, 0],corners[:, 1],corners[:, 2],corners[:, 3], ]).T
        # print(draw_corners.T, corners)
        # print(draw_corners, corners)
        center = _get_center(draw_corners)
        # print(center, corners)
        cframe = cv2.circle(cframe, (int(center[0]), int(center[1])), 10, [0, 255, 0], -1)

        print(f"X Err: {frame.shape[0]/2 - center[1]}")
        print(f"Y Err: {frame.shape[1]/2 - center[0]}")

        pts = np.concatenate((corners.T, np.ones([num_pts, 1])), axis=1)
        
        if show:
            # print(pts)
            for i in range(num_pts):
                cframe = cv2.circle(cframe, (int(pts[i][0]), int(pts[i][1])), 10, [255, 255, 0], -1)
            cframe = cv2.resize(cframe, (640, 640))

            cv2.imshow("test", cframe.astype(np.uint8))

            k = cv2.waitKey(1)
            if k%256 == 27:
                # ASCII:ESC pressed
                print("Escape hit, closing...")
                cv2.destroyAllWindows()
                break