import dynamixel_sdk as dxl
import time
import numpy as np
import math
import utils

from robot_addresses import *

# Connect to robot using the parameters defined in robot_addresses.py
portHandler = dxl.PortHandler(DEVICENAME)
packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)
portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)
for DXL_ID in DXL_IDS:
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_CW_COMPLIANCE_MARGIN, 0)
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_CCW_COMPLIANCE_MARGIN, 0)
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_CW_COMPLIANCE_SLOPE, 32)
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_CCW_COMPLIANCE_SLOPE, 32)
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_MOVING_SPEED, 100)

# Get camera matrix
K = utils.get_camera_matrix()

# Set a fitting angle for the robot to take a picture of the M&M's
goal_degrees = [0, 5, -90, -80] 
utils.set_goal_positions(portHandler, packetHandler, DXL_IDS, goal_degrees)

# Take picture with robot
img = utils.capture_image(save_img=False, show_img=False)

# Detect orange/reddish M&M's in image and return centers of the M&M's
center_values = utils.object_detector(img)

# Get transformation matrix
Ts_in0, _ = utils.our_forward_kinematics(math.radians(degree) for degree in goal_degrees)
T5_in0 = Ts_in0[-1]

A = np.array([1, 0, 0])
B = np.array([0, 1, 0])
C = np.array([0, 0, -40])

# Loop through all M&M's
for idx, center_value in enumerate(center_values):
    # Convert pixel coordinates to world coordinates                 
    point_reconstructed = utils.pixel_to_world(center_value, T5_in0, K, plane = (A, B, C), shape = np.array([640, 480]))
    point_reconstructed = np.array([[point_reconstructed[0]],[point_reconstructed[1]],[point_reconstructed[2]]])
    print(f'Reconstructed point: {point_reconstructed}')

    # Inverse kinematics on the reconstructed point
    thetas = utils.inverse_kinematics_our_robot(x4_in0_z=-1 ,o4_in0=point_reconstructed, point_outwards=True, elbow_up=True)

    # Convert radians to degrees
    goal_degrees = [np.degrees(theta) for theta in thetas]
    print(f'goal_degrees: {goal_degrees}')

    # If there is another M&M, set an intermidate point for the robot to travel to
    if idx+1 != len(center_values):
        # Convert pixel coordinates to world coordinates   
        point_reconstructed_next = utils.pixel_to_world(center_values[idx+1], T5_in0, K, plane = (A, B, C), shape = np.array([640, 480]))
        point_reconstructed_next = np.array([[point_reconstructed_next[0]],[point_reconstructed_next[1]],[point_reconstructed_next[2]]])
        # Take mean between the two points in world coordinates
        intermediate_point = (point_reconstructed+point_reconstructed_next)/2
        # Go up 40 mm in the intermediate point
        intermediate_point[-1] += 40
        # Inverse kinematics on the intermediate point
        thetas_intermediate = utils.inverse_kinematics_our_robot(x4_in0_z=-1 ,o4_in0=intermediate_point, point_outwards=True, elbow_up=True)
        # Convert radians to degrees
        goal_degrees_intermediate = [np.degrees(theta) for theta in thetas_intermediate]

    # Move robot to M&M
    utils.set_goal_positions(portHandler, packetHandler, DXL_IDS, goal_degrees)
    time.sleep(1)
    # Move robot to intermediate point if applicable
    if idx+1 != len(center_values):
        utils.set_goal_positions(portHandler, packetHandler, DXL_IDS, goal_degrees_intermediate)
