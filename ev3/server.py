#!/usr/bin/python
# RUN ON LAPTOP USING PYTHON 3.6

import socket
import time
from queue import Queue
import cv2
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import LKTracker
import time

# This class handles the Server side of the comunication between the laptop and the brick.
class Server:
    def __init__(self, hosts, ports):
       # setup server socket
        serversockets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM), socket.socket(socket.AF_INET, socket.SOCK_STREAM)] 
        # We need to use the ip address that shows up in ipconfig for the usb ethernet adapter that handles the comunication between the PC and the brick
        self.servers = [None, None]

        for i in range(len(hosts)):
            host, port = hosts[i], ports[i]

            print("Setting up Server\nAddress: " + str(host) + "\nPort: " + str(port))
            serversockets[i].bind((host, port))
            # queue up to 5 requests
            serversockets[i].listen(5) 
            self.servers[i], addr = serversockets[i].accept()
            print ("Connected to: " + str(addr))

    def sendSpeeds(self, speeds, queue):
        for i, server in enumerate(self.servers):
            # clamp the speed to the useable range
            print(speeds[i])
            for j in range(len(speeds[i])):
                speeds[i][j] = max(-2.5, min(speeds[i][j], 2.5))

            print(speeds[i], speeds[i][0])
            speed = ','.join(str(x) for x in speeds[i])
            print(speed)
            data = {"cmd": "MOVE", "data": speed}
            data = str(data)
            print(f"Sending Data: ({str(data)}) to robot. {i+1}")
            server.send(data.encode("UTF-8"))

            reply = server.recv(128).decode("UTF-8")
            queue.put(reply)
            assert queue.get(block=True) == "DONE" 

        
        

    # Sends a termination message to the client. This will cause the client to exit "cleanly", after stopping the motors.
    def sendTermination(self):
        data = {"cmd": "EXIT", "data": ""}
        data = str(data)
        for server in self.servers:
            server.send(data.encode("UTF-8"))

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


def move_left(speed):
    #    [A, B, C, D]
    C1 = np.array([1 * speed, 1 * speed, -1 * speed, -1 * speed])
    C2 = np.array([1 * speed, 0.5 * speed, 0, 0])
    
    return np.array([C1, C2])

def move_right(speed):
    #    [A, B, C, D]
    C1 = np.array([-1 * speed, -1 * speed, 1 * speed, 1 * speed])
    C2 = np.array([-.5 * speed, -1 * speed, 0, 0])
    
    return np.array([C1, C2])

def move_front(speed):
    #    [A, B, C, D]
    C1 = np.array([-1 * speed, 0.5 * speed, 0.5 * speed, -1 * speed])
    C2 = np.array([2 * speed, 2 * speed, 0, 0])
    
    return np.array([C1, C2])

def move_back(speed):
    #    [A, B, C, D]
    C1 = np.array([1 * speed, -0.25 * speed, -0.25 * speed, 1 * speed])
    C2 = np.array([-2 * speed, -2 * speed, 0, 0])
    
    return np.array([C1, C2])

def rise(speed):
    #    [A, B, C, D]
    C1 = np.array([-1 * speed, -1 * speed, -1 * speed, -1 * speed])
    C2 = np.array([-1 * speed, -1 * speed, 0, 0])
    
    return np.array([C1, C2])

def lower(speed):
    #    [A, B, C, D]
    C1 = np.array([1 * speed, 1 * speed, 1 * speed, 1 * speed])
    C2 = np.array([1 * speed, 1 * speed, 0, 0])
    
    return np.array([C1, C2])

def stop():
    #    [A, B, C, D]
    C1 = np.array([0, 0, 0, 0], dtype=np.float64)
    C2 = np.array([0, 0, 0, 0], dtype=np.float64)
    
    return np.array([C1, C2])



def connect(hosts, port = 9999):
    connected = False
    for host in hosts:
        try:
            server = Server(host, port)
            connected = True
            break
        except OSError:
            continue
    if connected:
        return server
    else:
        raise Exception("Could not connect to any of the listed hosts") 

def get_pos(frame, trackers):
    try:
        cframe = frame.copy()
    except AttributeError:
        cv2.destroyAllWindows()
        return 

    frame = cv2.medianBlur(frame,5)
    corners = np.zeros([2, num_pts])
    for i in range(num_pts):
        tracker = trackers[i]
        c = tracker.updateTracker(frame)
        corners[:, i] = _get_center(c)

    draw_corners = np.array([corners[:, 0],corners[:, 1],corners[:, 2],corners[:, 3], ]).T
    center = _get_center(draw_corners)

    print(center, frame.shape)
    y_err = frame.shape[0]/2 - center[1]
    x_err = frame.shape[1]/2 - center[0]
    print(f"X Err: {x_err}")
    print(f"Y Err: {y_err}")

    pts = np.concatenate((corners.T, np.ones([num_pts, 1])), axis=1)

    if show:
        # print(pts)
        for i in range(num_pts):
            cframe = cv2.circle(cframe, (int(pts[i][0]), int(pts[i][1])), 10, [255, 255, 0], -1)
            cframe = cv2.circle(cframe, (int(center[0]), int(center[1])), 10, [0, 255, 0], -1)
            cframe = cv2.circle(cframe, (int(frame.shape[1]/2), int(frame.shape[0]/2)), 5, [0, 0, 255], -1)
        cframe = cv2.resize(cframe, (640, 640))
    
    return center, cframe

def init_J(cam, trackers):
    speed = 15
    sleep_time = 1
    #      num err constraints x num motors
    J = np.zeros([2, 6])

    ret, frame = cam.read()
    init_pos, _ = get_pos(frame, trackers)
    

    for i in range(6):
        speeds = [[0, 0, 0, 0], [0, 0, 0, 0]]
        speeds[i // 4][i % 4] = speed
        server.sendSpeeds(speeds, queue)
        time.sleep(sleep_time)
        speeds = [[0, 0, 0, 0], [0, 0, 0, 0]]
        server.sendSpeeds(speeds, queue)

        ret, frame = cam.read()
        new_pos, _ = get_pos(frame, trackers)
        delta = new_pos-init_pos

        J[:, i] = [delta[1], delta[0]]
    
    J /= speed

    return J

def broyden(cam, trackers):
    ret, frame = cam.read()


    goal = [frame.shape[1]/2, frame.shape[0]/2]
    point, cframe = get_pos(frame, trackers)

    # The error vector. It is the vector from the point to the goalv
    error = [goal[0] - point[0],goal[1] - point[1]] 
    threshold = 5   # 25 pixels
    
    jacobian = init_J(cam, trackers)
    print(jacobian)
    inverse_jacobian = np.linalg.pinv(jacobian)
    idx = 0
    while np.linalg.norm(error) > threshold:
        ret, frame = cam.read()
        # We are following the formulae from the lab notes
        delta_angles = -np.matmul(inverse_jacobian, error) * 10
        print(delta_angles)
        speed0 = delta_angles[0:4]
        speed1 = np.zeros(4)
        speed1[0:2] = delta_angles[4:]
        print(speed0, speed1)
        server.sendSpeeds([speed0, speed1], queue)
        # time.sleep(2)  # Allow the motors to finish moving before we continue
        
        # Update the Jacobian every 5 steps to minimize updating the jacobian and recomputing its inverse every time
        # The reason we added this is because in general, it is expensive to update the jacobian and recomputing it's
        # inverse everytime, even though it doesn't matter that much in this case since this is a 2*2 Jacobian.
        # if idx % 50 == 0:
        #     print(error.shape, jacobian.shape, delta_angles.shape, (delta_angles.T * delta_angles).shape, np.outer((error - np.matmul(jacobian, delta_angles)) * (1 / np.dot(delta_angles.T, delta_angles)), delta_angles * 0.10).shape)
        #     jacobian = jacobian + np.outer((error - np.matmul(jacobian, delta_angles)) * (1 / np.dot(delta_angles.T, delta_angles)), delta_angles * .10)
        #     inverse_jacobian = np.linalg.pinv(jacobian)
        #     print(jacobian)
        point, cframe = get_pos(frame, trackers)
        error = [goal[0] - point[0],goal[1] - point[1]] 

        cv2.imshow("test", cframe.astype(np.uint8))
            
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ASCII:ESC pressed
            print("Escape hit, closing...")
            break
        idx += 1
    
    return error

MODE = "MANUAL"
# MODE = "PID"
thresh = 2
k_p = 1/10
if __name__ == "__main__":
    init_msg = f"# Initializing server in {MODE} mode... #"
    print("#" * len(init_msg))
    print(init_msg)
    print("#" * len(init_msg))
    
    host1 = "169.254.76.86"
    host2 = "169.254.247.244"
    port = 9999
    num_pts = 4
    show = True
    queue = Queue()

    hosts = ["169.254.76.86", "169.254.247.244"]
    ports = [9999, 9999]

    server = Server(hosts, ports)
    cam = cv2.VideoCapture(1)

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

    broyden(cam, trackers)

    # speeds = stop()
    # while True: 
    #     server.sendSpeeds(speeds, queue)
    #     # Capture frame-by-frame 
    #     ret, frame = cam.read() 
    #     try:
    #         cframe = frame.copy()
    #     except AttributeError:
    #         cv2.destroyAllWindows()
    #         break
    #     if not ret:
    #         break 

    #     frame = cv2.medianBlur(frame,5)
    #     corners = np.zeros([2, num_pts])
    #     for i in range(num_pts):
    #         tracker = trackers[i]
    #         c = tracker.updateTracker(frame)
    #         corners[:, i] = _get_center(c)
        
    #     draw_corners = np.array([corners[:, 0],corners[:, 1],corners[:, 2],corners[:, 3], ]).T
    #     center = _get_center(draw_corners)

    #     print(center, frame.shape)
    #     y_err = frame.shape[0]/2 - center[1]
    #     x_err = frame.shape[1]/2 - center[0]
    #     print(f"X Err: {x_err}")
    #     print(f"Y Err: {y_err}")

    #     pts = np.concatenate((corners.T, np.ones([num_pts, 1])), axis=1)
        
    #     if show:
    #         # print(pts)
    #         for i in range(num_pts):
    #             cframe = cv2.circle(cframe, (int(pts[i][0]), int(pts[i][1])), 10, [255, 255, 0], -1)
    #             cframe = cv2.circle(cframe, (int(center[0]), int(center[1])), 10, [0, 255, 0], -1)
    #             cframe = cv2.circle(cframe, (int(frame.shape[1]/2), int(frame.shape[0]/2)), 5, [0, 0, 255], -1)
    #         cframe = cv2.resize(cframe, (640, 640))

    #         cv2.imshow("test", cframe.astype(np.uint8))
            
    #     k = cv2.waitKey(1)

    #     if k%256 == 27:
    #         # ASCII:ESC pressed
    #         print("Escape hit, closing...")
    #         break
    #     if MODE == "MANUAL":
    #         # W
    #         if k == 119:
    #             # speeds = lower(5)
    #             speeds = move_front(5)
    #         # A
    #         elif k == 97:
    #             speeds = move_left(5)
    #         # S
    #         elif k == 115:
    #             speeds = move_back(5)
    #         # D
    #         elif k == 100:
    #             speeds = move_right(5)    
    #         else:
    #             speeds = stop()

    #     elif MODE == "PID":
    #         speeds = stop()
    #         if y_err > thresh:
    #             speeds += move_back(k_p * y_err)
    #         elif y_err < -thresh:
    #             speeds += move_front(k_p * y_err)
    #         print(k_p * y_err, speeds)

    #         if x_err < -thresh:
    #             speeds += move_right(k_p * x_err)
    #         elif x_err > thresh:
    #             speeds += move_left(k_p * x_err)
    #         print(k_p * x_err, speeds)
    #     time.sleep(0.1)
    
    server.sendSpeeds([[0, 0, 0, 0], [0, 0, 0, 0]], queue)
    server.sendTermination()

