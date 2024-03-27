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

	new_points = np.matmul(np.matmul(R_yaw, R_pitch), np.matmul(R_roll, object_points))
	# translate by the negative location of the camera platform

	translation = np.array([
		[-camera_location[0], -camera_location[0], -camera_location[0], -camera_location[0]],
		[-camera_location[1], -camera_location[1], -camera_location[1], -camera_location[1]],
		[-camera_location[2], -camera_location[2], -camera_location[2], -camera_location[2]],
	])

	new_points = new_points + translation

	# rotate around the camera platform

	R_yaw = np.array([
			[np.cos(-camera_location[3]), -np.sin(-camera_location[3]), 0],
			[np.sin(-camera_location[3]), np.cos(-camera_location[3]), 0],
			[0, 0, 1]
			])
	R_pitch = np.array([
			[np.cos(-camera_location[4]), 0, np.sin(-camera_location[4])],
			[0, 1, 0],
			[-np.sin(-camera_location[4]), 0, np.cos(-camera_location[4])]
			])
	R_roll = np.array([
			[1, 0, 0],
			[0, np.cos(-camera_location[5]), -np.sin(-camera_location[5])],
			[0, np.sin(-camera_location[5]), np.cos(-camera_location[5])]
			])
	
	new_points = np.matmul(np.matmul(R_yaw, R_pitch), np.matmul(R_roll, new_points))

	return new_points


def camera_projection(points):
	points = np.concatenate((points, np.ones((1,4))))
	f_matrix = np.array([
		[750, 0, 0],
		[0, 750, 0],
		[0, 0, 1]
		])
	camera_matrix = np.array([
		[1, 0, 0, 0],
		[0 ,1, 0 ,0],
		[0, 0, 1, 0]
	])
	projection_matrix = np.matmul(f_matrix, camera_matrix)
	projected_points = np.matmul(projection_matrix, points)
	projected_points = projected_points[:-1]/projected_points[-1]
	return projected_points

def compute_center_point(tracker_locations):
	p1 = np.array([tracker_locations[0][0], tracker_locations[1][0], 1])
	p2 = np.array([tracker_locations[0][1], tracker_locations[1][1], 1])
	p3 = np.array([tracker_locations[0][2], tracker_locations[1][2], 1])
	p4 = np.array([tracker_locations[0][3], tracker_locations[1][3], 1])

	l1 = np.cross(p1, p3)
	l2 = np.cross(p2, p4)

	center_point = np.cross(l1,l2)
	return center_point[:-1]/center_point[-1]

def compute_v(tracker_locations):
	p1 = np.array([tracker_locations[0][0], tracker_locations[1][0], 1])
	p2 = np.array([tracker_locations[0][1], tracker_locations[1][1], 1])
	p3 = np.array([tracker_locations[0][2], tracker_locations[1][2], 1])
	p4 = np.array([tracker_locations[0][3], tracker_locations[1][3], 1])

	l1 = np.cross(p1, p4)
	l2 = np.cross(p2, p3)

	v = np.cross(l1, l2)
	return v

def compute_h(tracker_locations):
	p1 = np.array([tracker_locations[0][0], tracker_locations[1][0], 1])
	p2 = np.array([tracker_locations[0][1], tracker_locations[1][1], 1])
	p3 = np.array([tracker_locations[0][2], tracker_locations[1][2], 1])
	p4 = np.array([tracker_locations[0][3], tracker_locations[1][3], 1])

	l1 = np.cross(p1, p2)
	l2 = np.cross(p4, p3)

	h = np.cross(l1, l2)
	return h

def compute_z_error(tracker_locations, targets):
	p1 = np.array([tracker_locations[0][0], tracker_locations[1][0], 1])
	p2 = np.array([tracker_locations[0][1], tracker_locations[1][1], 1])
	p3 = np.array([tracker_locations[0][2], tracker_locations[1][2], 1])
	p4 = np.array([tracker_locations[0][3], tracker_locations[1][3], 1])

	c3 = np.array([targets[0][0], targets[1][0], 1])
	c4 = np.array([targets[0][1], targets[1][1], 1])
	c1 = np.array([targets[0][2], targets[1][2], 1])
	c2 = np.array([targets[0][3], targets[1][3], 1])

	l1 = np.cross(p1, p4)
	l2 = np.cross(p2, p3)
	z_error = np.dot(c2, l1) + np.dot(c3, l1) + np.dot(c1, l2) + np.dot(c4, l2)
	return z_error





def main():
	# initialize positions
	# (x ,y ,z, yaw, pitch, roll)
	object_location = (0, 0, 20, 0.5, 0.3, 0)
	object_points = np.array([
		[-1, 1, 1, -1],
		[1, 1, -1, -1],
		[0, 0, 0, 0]
		])
	camera_location = (0, 0, 5, 0, 0, 0)

	# define target points (top left, top right, bot right, bot left)
	target_points = np.array([
		[-1, 1, 1, -1],
		[1, 1, -1, -1],
		[1, 1, 1, 1]
	])
	projected_target_points = camera_projection(target_points)
	

	obj_points_in_plf_frame = to_platform_frame(object_location, object_points, camera_location)
	projected_location = camera_projection(obj_points_in_plf_frame)

	# compute center point and lines
	center_point = compute_center_point(projected_location)
	x_error = - center_point[0]
	y_error = -center_point[1]
	v = compute_v(projected_location)
	yaw_error = v[0]
	roll_error = -v[2]
	h = compute_h(projected_location)
	pitch_error = h[2]
	z_error = compute_z_error(projected_location, projected_target_points)
	print(z_error)




	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(projected_location[0], projected_location[1])
	ax.scatter(projected_target_points[0], projected_target_points[1])
	ax.scatter(center_point[0], center_point[1])
	plt.show()





if __name__ == '__main__':
	main()
