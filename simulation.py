import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class PID():
    def __init__(self):
        self.integrals = np.array([0.0]*6)
        self.previous = np.array([None]*6)
        self.kp = np.array([5*10e-2, 5*10e-2, 10e-4, 1.5*10e-2, 1.0, 1.0])
        self.ki = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.kd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.first_update = True


    def update(self, errors):
        # compute p term
        p = np.multiply(self.kp, errors)
        # compute i term
        self.integrals = self.integrals + errors
        i = np.multiply(self.ki, self.integrals)
        # compute d term
        if self.first_update:
            self.previous = errors
            self.first_update = False
        d = errors - self.previous 
        d = np.multiply(self.kd, d)
        return p + i + d
        

        

def compute_camera_view(object_location, object_points, camera_location):
    obj_x, obj_y, obj_z, obj_yaw, obj_pitch, obj_roll = object_location
    cam_x, cam_y, cam_z, cam_yaw, cam_pitch, cam_roll = camera_location

    # object internal rotation
    obj_Rz = np.array([
        [np.cos(obj_yaw), -np.sin(obj_yaw), 0],
        [np.sin(obj_yaw), np.cos(obj_yaw), 0],
        [0, 0, 1]
    ])

    obj_Ry = np.array([
        [np.cos(obj_pitch), 0, np.sin(obj_pitch)],
        [0, 1, 0],
        [-np.sin(obj_pitch), 0, np.cos(obj_pitch)]
    ])

    obj_Rx = np.array([
        [1, 0, 0],
        [0, np.cos(obj_roll), -np.sin(obj_roll)],
        [0, np.sin(obj_roll), np.cos(obj_roll)]
    ])

    obj_R = np.matmul(obj_Rz, obj_Ry)
    obj_R = np.matmul(obj_R, obj_Rx)

    camera_centered_obj_points = np.matmul(obj_R, object_points)

    # object internal translation
    obj_trans = np.array([
        [obj_x, obj_x, obj_x, obj_x],
        [obj_y, obj_y, obj_y, obj_y],
        [obj_z, obj_z, obj_z, obj_z]
    ])

    camera_centered_obj_points = camera_centered_obj_points + obj_trans

    # external translation

    camera_trans = obj_trans = np.array([
        [cam_x, cam_x, cam_x, cam_x],
        [cam_y, cam_y, cam_y, cam_y],
        [cam_z, cam_z, cam_z, cam_z]
    ])

    camera_centered_obj_points = camera_centered_obj_points - camera_trans

    # external rotation

    cam_Rz = np.array([
        [np.cos(-cam_yaw), -np.sin(-cam_yaw), 0],
        [np.sin(-cam_yaw), np.cos(-cam_yaw), 0],
        [0, 0, 1]
    ])

    cam_Ry = np.array([
        [np.cos(-cam_pitch), 0, np.sin(-cam_pitch)],
        [0, 1, 0],
        [-np.sin(-cam_pitch), 0, np.cos(-cam_pitch)]
    ])

    cam_Rx = np.array([
        [1, 0, 0],
        [0, np.cos(-cam_roll), -np.sin(-cam_roll)],
        [0, np.sin(-cam_roll), np.cos(-cam_roll)]
    ])

    cam_R = np.matmul(cam_Rz, cam_Ry)
    cam_R = np.matmul(cam_R, cam_Rx)

    camera_centered_obj_points = np.matmul(cam_R, camera_centered_obj_points)

    focal_length = 750
    f_mat = np.array([
        [focal_length, 0, 0],
        [0, focal_length, 0],
        [0, 0, 1]
    ])

    c_mat = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])
    p_mat = np.matmul(f_mat, c_mat)
    camera_centered_obj_points = np.concatenate([camera_centered_obj_points, np.ones((1,4))])
    camera_centered_obj_points = np.matmul(p_mat, camera_centered_obj_points)


    return camera_centered_obj_points[:-1]/camera_centered_obj_points[-1]


def compute_errors(obj_points, target_points):
    p1 = np.array([obj_points[0][0], obj_points[1][0], 1])
    p2 = np.array([obj_points[0][1], obj_points[1][1], 1])
    p3 = np.array([obj_points[0][2], obj_points[1][2], 1])
    p4 = np.array([obj_points[0][3], obj_points[1][3], 1])

    t1 = np.array([target_points[0][0], target_points[1][0], 1])
    t2 = np.array([target_points[0][1], target_points[1][1], 1])
    t3 = np.array([target_points[0][2], target_points[1][2], 1])
    t4 = np.array([target_points[0][3], target_points[1][3], 1])

    # compute midpoint
    diag_1 = np.cross(p1, p3)
    diag_2 = np.cross(p2, p4)

    center = np.cross(diag_1, diag_2)
    center = center[:-1]/center[-1]
    x_error = center[0]
    y_error = center[1]
    
    # compute the vert lines
    vert_1 = np.cross(p1, p4)
    vert_2 = np.cross(p2, p3)

    # compute the vert vanishing point and yaw and pitch error
    vert_vanishing_point = np.cross(vert_1, vert_2)
    yaw_error = -vert_vanishing_point[0]/np.linalg.norm(vert_vanishing_point)
    pitch_error = vert_vanishing_point[2]/np.linalg.norm(vert_vanishing_point)
    
    # compute the horizontal lines
    hor_1 = np.cross(p1, p2)
    hor_2 = np.cross(p4, p3)

    # compute the horizontal vanishing point and roll error
    hor_vanishing_point = np.cross(hor_1, hor_2)
    roll_error = -hor_vanishing_point[2]/np.linalg.norm(hor_vanishing_point)
    
    
    # compute z error as the avg distance from the corners to the points
    true_avg_dist = np.sqrt(2*(750**2))
    dist_1 = np.sqrt((p1[0]-x_error)**2 + (p1[1]-y_error)**2)
    dist_2 = np.sqrt((p2[0]-x_error)**2 + (p2[1]-y_error)**2)
    dist_3 = np.sqrt((p3[0]-x_error)**2 + (p3[1]-y_error)**2)
    dist_4 = np.sqrt((p4[0]-x_error)**2 + (p4[1]-y_error)**2)
    avg_dist = (dist_1 + dist_2 + dist_3 + dist_4)/4
    z_error = true_avg_dist - avg_dist

    x_error = x_error/avg_dist
    y_error = y_error/avg_dist
    return np.array([x_error, y_error, z_error, yaw_error, roll_error, pitch_error])  

def main():
    pid = PID()
    #(x, y, z, yaw, roll, pitch)
    object_location = (0, 0, 20, 0, 0, 0)
    object_points = np.array([
        [-1, 1, 1, -1],
        [1, 1, -1, -1],
        [0, 0, 0, 0]
    ])
    camera_location = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
    target_points = np.array([
        [-1, 1, 1, -1],
        [1, 1, -1, -1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ])
    focal_length = 750
    f_mat = np.array([
        [focal_length, 0, 0],
        [0, focal_length, 0],
        [0, 0, 1]
    ])
    c_mat = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])
    p_mat = np.matmul(f_mat, c_mat)
    target_points = np.matmul(p_mat, target_points)
    target_points = target_points[:-1]/target_points[-1]

    for i in range(20):
        obj_in_cam_view = compute_camera_view(object_location, object_points, camera_location)
        errors = compute_errors(obj_in_cam_view, target_points)
        print(camera_location)
        print('------------------------')
        camera_location += pid.update(errors)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(obj_in_cam_view[0], obj_in_cam_view[1])
        ax.scatter(target_points[0], target_points[1])
        plt.show()






if __name__ == "__main__":
    main()