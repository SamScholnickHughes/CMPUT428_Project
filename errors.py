def compute_errors(obj_points):
    '''
        Compute the errors in all degrees of freedom
        
        Parameters:
            onj_points: np.array containing the location of the corners of the object in pixels,
            with the center of the image as the origin and in the order top_left, top_right, bottom_right, bottom_left
    '''
    p1 = np.array([obj_points[0][0], obj_points[1][0], 1])
    p2 = np.array([obj_points[0][1], obj_points[1][1], 1])
    p3 = np.array([obj_points[0][2], obj_points[1][2], 1])
    p4 = np.array([obj_points[0][3], obj_points[1][3], 1])


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
    # TODO set the true avg distance to the distance from the center of the object to 
    # one of the corners in pixels when the platform is in the correct location to pick
    # up the platform
    true_avg_dist = # np.sqrt(2*(750**2))
    dist_1 = np.sqrt((p1[0]-x_error)**2 + (p1[1]-y_error)**2)
    dist_2 = np.sqrt((p2[0]-x_error)**2 + (p2[1]-y_error)**2)
    dist_3 = np.sqrt((p3[0]-x_error)**2 + (p3[1]-y_error)**2)
    dist_4 = np.sqrt((p4[0]-x_error)**2 + (p4[1]-y_error)**2)
    avg_dist = (dist_1 + dist_2 + dist_3 + dist_4)/4
    z_error = true_avg_dist - avg_dist

    x_error = x_error/avg_dist
    y_error = y_error/avg_dist
    return np.array([x_error, y_error, z_error, yaw_error, roll_error, pitch_error])  
