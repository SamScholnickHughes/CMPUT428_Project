import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class StewartPlatform():
    def __init__(self, R, r):
        # origin in the center of all 
        self.base = np.array([
            [0, R, 0],
            [R*np.sqrt(3)/2, -R/2, 0],
            [-R*np.sqrt(3)/2, -R/2, 0]
            ]).T
            
        self.anchors = np.array([
            [0, -r, 0],
            [-r*np.sqrt(3)/2, r/2, 0],
            [r*np.sqrt(3)/2, r/2, 0]
            ]).T # these are coordinates of the anchors relative to the center of the platform
        
    def compute_lengths(self, x, y, z, roll, pitch, yaw):
        # compute the location of the anchors relative to the origin
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
            ])
        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
            ])
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
            ])
            
        Translation = np.array([
            [x, x, x],
            [y, y, y],
            [z, z, z]
            ])
       
       
        points = np.matmul(np.matmul(R_yaw, R_pitch), np.matmul(R_roll, self.anchors)) + Translation
        
        # compute the distances between each base point and anchor
        # note anchor i is connected to base points (i+1) mod 3 and (i+2) mod 3 
        # and vice versa base point j is connected to anchors (j+1) mod 3 and (j+2) mod 3
        
        Ab = np.linalg.norm(self.base[:,0] - points[:,1])
        Ac = np.linalg.norm(self.base[:,0] - points[:,2])
        
        Ba = np.linalg.norm(self.base[:,1] - points[:,0])
        Bc = np.linalg.norm(self.base[:,1] - points[:,2])
        
        Ca = np.linalg.norm(self.base[:,2] - points[:,0])
        Cb = np.linalg.norm(self.base[:,2] - points[:,1])
        
        return (Ab, Ac, Ba, Bc, Ca, Cb)
       
       
       


def main():
	platform = StewartPlatform(10, 3)
	print(platform.compute_lengths(1,0,3,0,0,0))






if __name__ == '__main__':
    main()
