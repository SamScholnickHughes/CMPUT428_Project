#!/usr/bin/python3       
# RUN ON BRICK
    
import socket
import os
import time
import ast
from ev3dev2.motor import LargeMotor

from ev3dev2.motor import OUTPUT_A, OUTPUT_B, OUTPUT_C, OUTPUT_D
# from util import ArmMotor

# This class handles the client side of communication. It has a set of predefined messages to send to the server as well as functionality to poll and decode data.
class Client:
    def __init__(self, host, port):
        # We need to use the ipv4 address that shows up in ipconfig in the computer for the USB. Ethernet adapter handling the connection to the EV3
        print("Setting up client\nAddress: " + host + "\nPort: " + str(port))
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        self.s.connect((host, port))  
        self.done = False

        self.first_motor = LargeMotor(OUTPUT_A)
        self.second_motor = LargeMotor(OUTPUT_B)
        self.third_motor = LargeMotor(OUTPUT_C)
        self.fourth_motor = LargeMotor(OUTPUT_D)
            
        
    # Block until a message from the server is received. When the message is received it will be decoded and returned as a string.
    # Output: UTF-8 decoded string containing the instructions from server.5
    def pollData(self):
        print("Waiting for Data")
        data = self.s.recv(128).decode("UTF-8")
        print("Data Received")
        print(data)
        if data != "":
            return ast.literal_eval(data)
        else:
            return {"cmd": "404"}
    
    def run_cmd(self, cmd, data):
        if cmd == "EXIT":
            self.get_done(data)
        elif cmd == "MOVE":
            self.move_cmd(data)
        elif cmd == "GET_ANGLES":
            self.get_angles(data)
        else:
            print("404")
    
    # Sends a message to the server letting it know that the moveme5nt of the motors was executed without any inconvenience.
    def sendDone(self):
        self.s.send("DONE".encode("UTF-8"))

    # Sends a message to the server letting it know that there was an isse during the execution of the movement (obstacle avoided) and that the initial jacobian should be recomputed (Visual servoing started from scratch)
    def sendReset(self):
        self.s.send("RESET".encode("UTF-8"))

    def move_cmd(self, data):
        speeds = [float(speed) for speed in data.split(",")]
        self.first_motor.on(speeds[0])
        self.second_motor.on(speeds[1])
        self.third_motor.on(speeds[2])
        self.fourth_motor.on(speeds[3])
        self.sendDone()

    def get_angles(self, data):
        res = "[{}, {}]".format(self.second_motor.position, self.second_motor.position)
        self.s.send(res.encode("UTF-8"))
        self.sendDone()


    def get_done(self, data):
        self.first_motor.on(0)
        self.second_motor.on(0)
        self.third_motor.on(0)
        self.fourth_motor.on(0)
        self.done = True
        self.sendDone()

def connect(hosts, port = 9999):
    connected = False
    for host in hosts:
        try:
            client = Client(host, port)
            connected = True
            break
        except OSError:
            continue
    if connected:
        return client
    else:
        raise Exception("Could not connect to any of the listed hosts") 

host = "169.254.76.86"
hosts = ["169.254.76.86", "169.254.247.244"]
port = 9999
client = connect(hosts)
i = 0
DONE = False

print("Client hosted at {host}, on port {port}")
while not client.done:
    data = client.pollData()
    client.run_cmd(data["cmd"], data["data"])

print("CLIENT HAS STOPPED")
