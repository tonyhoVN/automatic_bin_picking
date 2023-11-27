import rospy
import numpy as np
from math import sin,cos,tan,pi,atan2, sqrt
import os, sys, time
from numpy.linalg import inv
from tf2_msgs.msg import geometry_msgs, TFMessage
import tf2_ros
from tf import transformations
from scipy.spatial.transform import Rotation as R

# Setup robot
sys.dont_write_bytecode = True
sys.path.append( os.path.abspath(os.path.join(os.path.dirname(__file__),"../../doosan-robot/common/imp")) ) # get import path : DSR_ROBOT.py 

ROBOT_ID     = "dsr01"
ROBOT_MODEL  = "m1013"
import DR_init
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
from DSR_ROBOT import *

# Global setup
set_velx(150)  # set global task speed: 30(mm/sec), 20(deg/sec)
set_accx(300)  # set global task accel: 60(mm/sec^2), 40(deg/sec^2)
set_velj(30)  # set global joint speed: 10 deg/s
set_accj(60)  # set global joint accel: 20 deg/s^2
set_robot_mode(0) # set control mode to manual 

# HOME Position
HOME = [0.890407919883728, 
        -2.1886579990386963, 
        104.82962799072266, 
        0.00029605263262055814, 
        77.35381317138672, 
        0.8971874713897705] # Joint Home

# HOMEX = list(fkin(HOME,ref=DR_BASE)) # Cartesian Home
HOMEX = [600.3831176757812, 70.744873046875, 630.87139892578125, 0.0, 180.0, 0.0]
movel(HOMEX)

# Target bin  
TARGET_BIN = [281.0, 98.3, 406.0, 0.0, 180.0, 0.0]

## Transfromation matrix 
Z_T_E = 170 #mm - distans from ee to tool

H_T_C = np.array([[1, 0, 0, -35.2988],  #34.07
                  [0, 1, 0, -60.3], #52.7
                  [0, 0, 1, 95], #28
                  [0, 0, 0, 1]]) ## transformation matrix from tool to cam coordinate (mm)

H_C_T = np.array([[1, 0, 0, 35.2988],
                  [0, 1, 0, 60.3],
                  [0, 0, 1, -95],
                  [0, 0, 0, 1]]) ## transformation matrix from cam to tool coordinate (mm)


H_B_S = np.eye(4) ## from cam to scene coordinate (mm)
H_S_O = np.eye(4) ## from scene to center object
H_B_T = np.eye(4) ## from base to tool-flage 
H_B_C = np.eye(4) ## from base to cam 
H_T_E = np.eye(4) ## from tool-flage to end tool 
H_T_E[0:3, 3] = [0,0,Z_T_E]
H_E_T = [[1, 0, 0, 0], 
         [0, 1, 0, 0],
         [0, 0, 1, -Z_T_E],
         [0, 0, 0, 1]]

H_B_R = np.eye(4) ## base-reference 
H_B_O = np.eye(4) ## base-object

## Transform frame 
base_frame = "base_0"
tool_frame = "link6"
camera_frame = "camera_color_optical_frame"
end_tool_frame = "end_tool"
target_object_frame = "object"
scene_frame = "scene"

broadcaster_static = tf2_ros.StaticTransformBroadcaster()
broadcaster = tf2_ros.TransformBroadcaster()

def rot_to_zyz(matrix):
    rotation_matrix = matrix[:3,:3]
    r = R.from_matrix(np.array(rotation_matrix))
    theta3, theta2, theta1 = r.as_euler('zyz', degrees=True)
    
    return theta1, theta2, theta3


#### Moving Functions ####

def public_tf(type: str, H_list, frame1_list, frame2_list):
    
    transform_stamped_list = []
    for (H, frame1, frame2) in zip(H_list, frame1_list, frame2_list):
        # Create a transform message 
        transform_stamped = geometry_msgs.msg.TransformStamped()
        transform_stamped.header.stamp = rospy.Time.now()
        transform_stamped.header.frame_id = frame1
        transform_stamped.child_frame_id = frame2

        # Extract the translation and rotation from the transformation matrix
        translation = H[0:3, 3]/1000 ## mm to m
        rotation_matrix = np.eye(4)
        rotation_matrix[0:3, 0:3] = H[0:3, 0:3]

        # Convert the rotation matrix to a quaternion
        rotation = transformations.quaternion_from_matrix(rotation_matrix)
        
        # Set the translation and rotation in the transform message
        transform_stamped.transform.translation.x = translation[0]
        transform_stamped.transform.translation.y = translation[1]
        transform_stamped.transform.translation.z = translation[2]
        transform_stamped.transform.rotation.x = rotation[0]
        transform_stamped.transform.rotation.y = rotation[1]
        transform_stamped.transform.rotation.z = rotation[2]
        transform_stamped.transform.rotation.w = rotation[3]

        transform_stamped_list.append(transform_stamped)

    # Publish tf
    if type == "static":
        broadcaster_static.sendTransform(transform_stamped_list)
    elif type == "normal":
        broadcaster.sendTransform(transform_stamped_list)
    else:
        rospy.loginfo("WRONG TYPE")
        pass
    
