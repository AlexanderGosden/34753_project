import numpy as np

#DH convention for our robot
d1 = 47 #distance from coordinate frame 0 to 1 (in mm), along z-axis of coordinate frame 0 (50)
a2 = 93 #distance from coordinate frame 1 to 2 (in mm), along x-axis of coordinate frame 2
a3 = 93 #distance from coordinate frame 2 to 3 (in mm), along x-axis of coordinate frame 3
a4 = 72 #distance from coordinate frame 3 to 4 (in mm), along x-axis of coordinate frame 4 (0) (85 mm from frame 3 to 4)
joint_types = ["R", "R", "R", "R"] #all revolute

our_DH = lambda theta1, theta2, theta3, theta4 : [[theta1, d1, 0, np.pi/2], [np.pi/2 + theta2, 0, a2, 0], [theta3, 0, a3, 0], [theta4, 0, a4, 0]]