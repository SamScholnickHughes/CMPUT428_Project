import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def to_platform_frame(object_location, object_points, camera_location):
	# first rotate the points of the object
	R_yaw = np.array([
			[np.cos(object_location[3]), -np.sin(object_location[3]), 0],
			[np.sin(object_location[3]), np.cos(object_location[3]), 0],
			[0, 0, 1]
			])
		R_pitch = np.array([
			[np.cos(object_location[4]), 0, np.sin(object_location[4])],
			[0, 1, 0],
			[-np.sin(object_location[4]), 0, np.cos(object_location[4])]
			])
		R_roll = np.array([
			[1, 0, 0],
			[0, np.cos(object_location[5]), -np.sin(object_location[5])],
			[0, np.sin(object_location[5]), np.cos(object_location[5])]
			])

		# translate by the negative location of the camera platform
		

def main():
	# (x ,y ,z, yaw, pitch, roll)
	object_location = (0 ,0, 20, 0, 0, 0)
	object_points = np.array([
		[-1, 1, 1, -1],
		[1, 1, -1, -1],
		[0, 0, 0, 0]
		])
	camera_location = (0, 0, 5, 0, 0, 0)





if __name__ == '__main__':
	main()
