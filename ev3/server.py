#!/usr/bin/python
# RUN ON LAPTOP USING PYTHON 3.6

import socket
import time
from queue import Queue
import json

# This class handles the Server side of the comunication between the laptop and the brick.
class Server:
    def __init__(self, host, port):
       # setup server socket
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        # We need to use the ip address that shows up in ipconfig for the usb ethernet adapter that handles the comunication between the PC and the brick
        print("Setting up Server\nAddress: " + str(host) + "\nPort: " + str(port))
        
        serversocket.bind((host, port))
        # queue up to 5 requests
        serversocket.listen(5) 
        self.cs, addr = serversocket.accept()
        print ("Connected to: " + str(addr))

    # Sends set of angles to the brick via TCP.
    # Input: base_angle [Float]: The angle by which we want the base to move
    #        joint_angle [Float]: The angle by which we want to joint to move
    #        queue [Thread-safe Queue]: Mutable data structure to store (and return) the messages received from the client
    def sendSpeeds(self, speeds, queue):
        # Format in which the client expects the data: "angle1,angle2"
        data = {"cmd": "MOVE", "data": str(speeds[0]) + "," + str(speeds[1])}
        data = str(data)
        # data = str(base_angle) + "," + str(joint_angle)
        print("Sending Data: (" + str(data) + ") to robot.")
        self.cs.send(data.encode("UTF-8"))
        # Waiting for the client (ev3 brick) to let the server know that it is done moving
        reply = self.cs.recv(128).decode("UTF-8")
        queue.put(reply)
        assert queue.get(block=True) == "DONE" 

    # Sends a termination message to the client. This will cause the client to exit "cleanly", after stopping the motors.
    def sendTermination(self):
        data = {"cmd": "EXIT", "data": ""}
        data = str(data)
        self.cs.send(data.encode("UTF-8"))

    def getAngles(self):
        data = {"cmd": "GET_ANGLES", "data": ""}
        data = str(data)
        self.cs.send(data.encode("UTF-8"))
        reply = json.loads(self.cs.recv(128).decode("UTF-8"))
        print(reply)
        return reply

    # Lets the client know that it should enable safety mode on its end
    def sendEnableSafetyMode(self):
        self.cs.send("SAFETY_ON".encode("UTF-8"))
    
    # Lets the client know that it should disable safety mode on its end
    def sendDisableSafetyMode(self):
        self.cs.send("SAFETY_OFF".encode("UTF-8"))


if __name__ == "__main__":
    host = "169.254.79.135"
    port = 9999
    server = Server(host, port)
    queue = Queue()

    server.sendSpeeds([10, 10], queue)
    time.sleep(2.5)
    server.sendSpeeds([0, 0], queue)
    server.sendTermination()