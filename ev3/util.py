# This file contains utility functions and classes used in both questions of the lab
# to avoid code repetition.

from ev3dev2.motor import LargeMotor
from math import radians, cos, sin, sqrt

# All length units are in cm, all angles are in degrees

l1, l2 = 11, 7

class ArmMotor(LargeMotor):
    def __init__(self, OUTPUT, block=True):
        super(ArmMotor, self).__init__(OUTPUT)
        self.initial_pos = self.position
        self.STOP_ACTION_HOLD = "brake" # make it so that the motors can be moved by hand
        self.block = block
        
    def move_angle(self, theta, spd=10):
        self.on_for_degrees(spd, theta, block=self.block)

    def reset(self):
        self.move_angle(-self.position) 
        # self.move_angle(self.initial_pos-self.position) 

    def calibrated_position(self):
        return self.position
        #return super().position-self.initial_pos
    
    def __str__(self) -> str:
        return str(self.calibrated_position())
    
def calculate_coordinates(theta1, theta2):
    theta1, theta2 = radians(theta1), radians(theta2)
    x = l1*cos(theta1) + l2*cos(theta1+theta2)
    y = l1*sin(theta1) + l2*sin(theta1+theta2)
    return [x, y]