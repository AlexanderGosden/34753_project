import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

# Import parameters for robot
from parameters import *
from robot_addresses import *


# Rotation matrices
Rx = lambda theta : np.array([[1,           0,              0],
                              [0,           np.cos(theta),  -np.sin(theta)],
                              [0,           np.sin(theta),  np.cos(theta)]])

Ry = lambda theta : np.array([[np.cos(theta),  0,  np.sin(theta)],
                                [0,              1,  0],
                                [-np.sin(theta), 0,  np.cos(theta)]])

Rz = lambda theta : np.array([[np.cos(theta),  -np.sin(theta), 0],
                                [np.sin(theta),  np.cos(theta),  0],
                                [0,              0,              1]])


# Function to homogenize a rotation matrix
hom_R = lambda R : np.concatenate([np.concatenate((R, np.array([[0, 0, 0]]).T), axis = 1),
                                    np.array([[0, 0, 0, 1]])], axis = 0)
# Homogeneous rotation matrices
R_x_hom = lambda theta : hom_R(Rx(theta))
R_y_hom = lambda theta : hom_R(Ry(theta))
R_z_hom = lambda theta : hom_R(Rz(theta))

# Homogeneous translation matrices
Tx = lambda d : np.array([[1, 0, 0, d],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
Ty = lambda d : np.array([[1, 0, 0, 0],
                          [0, 1, 0, d],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
Tz = lambda d : np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, d],
                          [0, 0, 0, 1]])

# Denavit-Hartenberg transformation
T_DH = lambda theta, d, a, alpha : R_z_hom(theta) @ Tz(d) @ Tx(a) @ R_x_hom(alpha)


# Function to get the transformation matrices given a set of DH params
def forward_kinematics(DH_params, ax = None, linecolor = 'b'):

    T = np.eye(4)
    Ts = [T]
    Ts_in0 = [T]

    for theta, d, a, alpha in DH_params:
        next_T = T_DH(theta, d, a, alpha)
        T = T @ next_T

        if ax is not None:
            o = T[:3, 3]
            ax.scatter(*o.flatten())

        Ts.append(next_T)
        Ts_in0.append(T)

    # Last x-axis
    if ax is not None:
        ax.quiver(*o.flatten(), *T[:3, 0].flatten()*20, color = 'r')
        ax.plot3D(*np.array([T[:3, 3] for T in Ts_in0]).T, color = linecolor)

    return Ts_in0, Ts


def our_forward_kinematics(thetas):
    # Calculates the forward kinematics for the DH frames + for frame 5
    
    DH = our_DH(*thetas)
    Ts_in0, Ts = forward_kinematics(DH)
    
    T_5_in_4 = Tx(-5) @ Ty(25) # Translation from frame 4 to frame 5
    T_5_in_0 = Ts_in0[-1] @ T_5_in_4
    
    Ts.append(T_5_in_4)
    Ts_in0.append(T_5_in_0)
    
    return Ts_in0, Ts


def set_goal_positions(portHandler, packetHandler, DXL_IDS, goal_degrees):
    """
    Sets each motor to the specified goal position and checks the actual position.

    Parameters:
    - portHandler: The port handler object for managing the connection.
    - packetHandler: The packet handler object for communication.
    - DXL_IDS: List of motor IDs to control.
    - goal_degrees: List of goal positions (in degrees, -150 to +150) corresponding to each motor.
    """
    def degrees_to_motor_position(degrees):
        # Ensure the degrees are within the valid range
        if degrees < -150 or degrees > 150:
            raise ValueError("Degrees must be between -150 and +150")

        # Convert degrees to motor position
        motor_position = int((degrees + 150) / 300 * 1023)
        return motor_position
    
    def motor_position_to_degrees(motor_position):
        # Ensure the motor position is within the valid range
        if motor_position < 0 or motor_position > 1023:
            raise ValueError("Motor position must be between 0 and 1023")

        # Convert motor position to degrees
        degrees = (motor_position / 1023) * 300 - 150
        return degrees

    # Convert goal positions from degrees to motor position range
    goal_positions = [degrees_to_motor_position(angle) for angle in goal_degrees]

    # Ensure the goal positions list matches the number of motors
    if len(DXL_IDS) != len(goal_positions):
        print("Error: Mismatch between DXL_IDS and goal_positions length.")
        return

    # Set each motor to its goal position
    for i, DXL_ID in enumerate(DXL_IDS):
        # Write goal position to each motor
        packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_GOAL_POSITION, goal_positions[i])
        print(f"Motor {DXL_ID} set to position {goal_degrees[i]} degrees")

    # Wait until all motors reach their positions (±2 degrees tolerance)
    while True:
        all_motors_in_position = True
        for i, DXL_ID in enumerate(DXL_IDS):
            dxl_present_position, _, _ = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_MX_PRESENT_POSITION)
            dxl_present_degrees = motor_position_to_degrees(dxl_present_position)
            if abs(dxl_present_degrees - goal_degrees[i]) > 2:
                all_motors_in_position = False
        if all_motors_in_position:
            break
        time.sleep(0.1)  # Short delay to prevent excessive polling

    # Final check of motor positions
    for i, DXL_ID in enumerate(DXL_IDS):
        dxl_present_position, _, _ = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_MX_PRESENT_POSITION)
        print(f"Motor {DXL_ID} reached position {motor_position_to_degrees(dxl_present_position)} degrees")


i = 0  # Global counter

def capture_image(save_img=False, show_img=False):
    global i  # Use the global variable

    # Open video device (assuming index 1 is the robot camera)
    capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not capture.isOpened():
        print("Error: Could not open video device.")
        return

    # Capture an image
    ret, img = capture.read()
    if ret:
        # Convert the image to RGB format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if save_img:
            # Save the image with OpenCV
            filename = f'image{i}.png'
            cv2.imwrite(filename, img_rgb)
            print(f"Image saved as {filename}")
            i += 1
        
        if show_img:
            # Display the image inline using Matplotlib
            plt.figure(figsize=(6, 4))
            plt.imshow(img_rgb)
            plt.axis('off')  # Hide axes for a cleaner look
            plt.title(f'Captured Image {i}')
            plt.show()
    else:
        print(f"Failed to capture image {i}.")

    # Release the camera
    capture.release()
    return img


def get_camera_matrix():
    images = []

    # Load images
    for i in range(0, 13):  # From 1 to 10
        filename = f"image{i}.png"
        img = cv2.imread(filename)  # Read image
        if img is not None:  # Check if the image was successfully loaded
            images.append(img)

    # Define the checkerboard dimensions (7x9, squares are 20x20 mm)
    checkerboard_dims = (7, 9)
    square_size = 20  # in mm

    # Prepare object points (3D points in real world space)
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane

    # Generate the 3D object points for the checkerboard (assuming z = 0)
    obj_p = np.zeros((checkerboard_dims[0] * checkerboard_dims[1], 3), np.float32)
    obj_p[:, :2] = np.indices(checkerboard_dims).T.reshape(-1, 2)
    obj_p *= square_size  # Scale by the square size in mm

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_dims, None)
        
        if ret:
            img_points.append(corners)
            obj_points.append(obj_p)
            
            # Draw the corners on the image
            #cv2.drawChessboardCorners(img, checkerboard_dims, corners, ret)
            #cv2.imshow('Corners', img)
            #cv2.waitKey(500)  # Display each image for 500ms

    cv2.destroyAllWindows()

    # Perform camera calibration
    ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None)

    return K

def world_to_pixel(Q, T_5_in0, K, shape = np.array([640, 480])):
    #Q is a point in world coordinates
    #shape is width x height, i.e. x times y
    
    #transform to homogeneous world coordinates
    Q_hom = np.array([*Q, 1])
    
    #transform to camera coordinates with x pointing out of the camera
    p_star_xout_hom = np.linalg.inv(T_5_in0) @ Q_hom
    
    #transform to camera coordinates with z pointing out of the camera (swap x and z). This is necessary so we can use the camera matrix
    p_star_hom = p_star_xout_hom
    p_star_hom[[0, 2]] = p_star_hom[[2, 0]]
    
    #make non-homogeneous. At this point, the point is a specific 3D point in the camera frame
    p_star = p_star_hom[:3] / p_star_hom[3]
    
    #transform to a homogeneous representation of a 2D point in pixel coordinates.
    # Because it's homogeneous, it cannot be distuingished from any other point on the same ray from the camera, so can be interpreted as a direction
    pixel_hom = K @ p_star
    
    #make non-homogeneous
    pixel = pixel_hom[:2] / pixel_hom[2]
    
    #flip the pixel coordinates. TODO: Check if we should flip x and y, or just y
    flipped = [pixel[0], shape[1]-pixel[1]]
    return flipped

def pixel_to_world(flipped, T_5_in0, K, plane, shape = np.array([640, 480])):
    #input coors should be the flipped coordinates, like returned by to_pixel_coordinates
    #plane is (A, B, C)-tuple, each must have shape (3,)
    
    #extract the plane
    A, B, C = plane
    #calculate the normal vector
    n = np.cross(A, B)
    
    
    #unflip the pixel coordinates
    pixel = [flipped[0], shape[1]-flipped[1]]

    #make homogeneous
    pixel_hom = np.array([*pixel, 1])
    
    #transform to a direction in camera coordinates.
    r_c = np.linalg.inv(K) @ pixel_hom #should be 1/k * pstar
    
    #make the x-axis be the one that points out of the camera
    r_c_xout = r_c
    r_c_xout[[0, 2]] = r_c_xout[[2, 0]]
    
    #transform r_c to world coordinates.
    
    t = T_5_in0[:3, 3] #extract translation
    R = T_5_in0[:3, :3] #extract rotation. Because of our setup, this rotates for the x-axis pointing out of the camera
    
    r_w = R @ r_c_xout
    
    #factor, using the plane equation, i.e. p = t + k*r_w is in the plane iff n*(p - C) = 0
    k = np.dot(n, C - t) / np.dot(n, r_w)
    
    #point
    point = t + k*r_w
    
    return point

def object_detector(img):
    # Convert the image from BGR to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the HSV range for orange color
    lower_orange = np.array([0, 100, 80])  # Lower bound for orange
    upper_orange = np.array([15, 255, 200])  # Upper bound for orange

    # Create a mask for orange areas
    orange_mask = cv2.inRange(img_hsv, lower_orange, upper_orange)

    # Find contours in the orange mask
    contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Copy the original image to draw on
    img_with_circles = img.copy()

    # Define the minimum radius threshold (e.g., 10 pixels)
    min_radius = 10

    center_values = []

    # Loop through each contour
    for contour in contours:
        # Get the minimum enclosing circle for each contour
        (x, y), radius = cv2.minEnclosingCircle(contour)

        # Skip small circles
        if radius < min_radius:
            continue
        
        center_values.append([x, y])
        # Draw the circle around the detected orange area
        cv2.circle(img_with_circles, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    # Convert the result to RGB for displaying with matplotlib
    img_with_circles_rgb = cv2.cvtColor(img_with_circles, cv2.COLOR_BGR2RGB)

    # Display the result with circles
    plt.imshow(img_with_circles_rgb)
    if center_values:
        plt.scatter(center_values[0][0], center_values[0][1], color='blue')
    plt.axis('off')  # Hide the axis
    plt.savefig('Object_detection.png')
    
    return center_values


def inverse_kinematics_our_robot(x4_in0_z, o4_in0, lean_back = False, elbow_up = False, point_outwards = False):
    x, y, z = o4_in0.flatten()
    
    
    if lean_back:
        theta1 = np.arctan2(y, x) + np.pi
    else:
        theta1 = np.arctan2(y, x)
        
    magnitude_for_first_two_coors = np.sqrt(1 - x4_in0_z**2)
    
    
    x4_in0_x = magnitude_for_first_two_coors*np.cos(theta1)
    x4_in0_y = magnitude_for_first_two_coors*np.sin(theta1)
    
    # Should the styles point away from the robot or not?
    if (point_outwards and lean_back) or (not point_outwards and not lean_back):
        x4_in0_x *= -1
        x4_in0_y *= -1
        
    x4_in0 = np.array([x4_in0_x, x4_in0_y, x4_in0_z]).reshape(3, 1)
    
    
    
    o3_in0 = o4_in0 - a4*x4_in0 # Position of coordinate frame 3
    
    xc, yc, zc = o3_in0.flatten()
    
    r = np.sqrt(xc**2 + yc**2)
    s = zc - d1
    d = np.sqrt(r**2 + s**2)

    
    cos_3 = -(a2**2 + a3**2 - d**2)/(2*a2*a3)
    cos_3 = np.clip(cos_3,-1,1)
    if (elbow_up and not lean_back) or (not elbow_up and lean_back):
        sin_3 = -np.sqrt(1 - cos_3**2)
    else:
        sin_3 = np.sqrt(1 - cos_3**2)
        
    
    theta3 = np.arctan2(sin_3, cos_3)
    
    phi1_abs = np.arctan2(s, r)
    phi2_abs = np.arctan2(a3*np.abs(sin_3), a2 + a3*cos_3)
    
    if (elbow_up and not lean_back):
        theta2 = -np.pi/2 + phi1_abs + phi2_abs
    elif (elbow_up and lean_back):
        theta2 = np.pi/2 - phi1_abs - phi2_abs
    elif (not elbow_up and not lean_back):
        theta2 = -np.pi/2 + phi1_abs - phi2_abs
    elif (not elbow_up and lean_back):
        theta2 = np.pi/2 - phi1_abs + phi2_abs
        
    if (point_outwards and not lean_back) or (not point_outwards and lean_back):
        pointer_in_xy_direction = np.sqrt(x4_in0_x**2 + x4_in0_y**2)
    else:
        pointer_in_xy_direction = -np.sqrt(x4_in0_x**2 + x4_in0_y**2)
    
    
    
    goal_angle = np.arctan2(x4_in0_z, pointer_in_xy_direction).squeeze()
    theta4 = goal_angle - np.pi/2 - theta2 - theta3
    
    return theta1, theta2, theta3, theta4



"""
def inverse_kinematics_our_robot_old(x4_in0_z, o4_in0):
    x, y, z = o4_in0.flatten()
    theta1 = np.arctan2(y, x)

    magnitude_for_first_two_coors = np.sqrt(1 - x4_in0_z**2)
    x4_in0_x = magnitude_for_first_two_coors*np.cos(theta1)
    x4_in0_y = magnitude_for_first_two_coors*np.sin(theta1)

    x4_in0 = np.array([x4_in0_x, x4_in0_y, x4_in0_z]).reshape(3, 1)

    ##decoupling step 1: Calculate first where the coordinate frame 3 is
    o3_in0 = o4_in0 - a4*x4_in0 #position of coordinate frame 3
    xc, yc, zc = o3_in0.flatten()

    ##decoupling step 2: Calculate the joint angles to get to frame 3
    # Calculate theta1 geometrically

    # Calculate theta3 geometrically
    r = np.sqrt(xc**2 + yc**2)
    s = zc - d1

    cos_3 = -(a2**2 + a3**2 - r**2 - s**2)/(2*a2*a3) # Cosine of theta3, according to law of cosines

    cos_3 = np.clip(cos_3, -1, 1) # Numerical errors can make cos_3 slightly out of bounds

    sin_3 = np.sqrt(1 - cos_3**2) # Plus/minus sine of theta3, from idiot rule. Here we chose elbow down optionn

    theta3 = np.arctan2(sin_3, cos_3) # Chose elbow down position using sign

    # Calculate theta2 geometrically
    theta2 = np.arctan2(s, r) - np.arctan2(a3*sin_3, a2 + a3*cos_3) - np.pi/2

    # Angle of the stylus relative to world level
    goal_angle = np.arctan2(x4_in0[2], np.sqrt(x4_in0[0]**2 + x4_in0[1]**2)).squeeze()
    theta4 = goal_angle - np.pi/2 - theta2 - theta3


    return theta1, theta2, theta3, theta4
"""