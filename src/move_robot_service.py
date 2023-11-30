#!/usr/bin/env python3
from Robot_Setup import *

## Moving function
def move_to_scan_position(point: list, radius: float):
    '''
    Service that moves robot close to object 
    REMEMBER: coordinate in milimeter
    '''

    point_object_cam = list(point) # distance from object to depth camera  

    # Calculate distance from object to tool flage
    point_object_tool = [0,0,0]
    point_object_tool[0] = point_object_cam[0] + H_T_C[0][3] 
    point_object_tool[1] = point_object_cam[1] + H_T_C[1][3]
    point_object_tool[2] = point_object_cam[2] + H_T_C[2][3]
    
    rospy.loginfo("Come to object, start scanning")
    # Move to position above box
    # target1 = [point_object_cam[0], point_object_cam[1], (1/3)*float(point_object_cam[2]), 0, 0, 0]
    target1 = [point_object_cam[0], point_object_cam[1], point_object_cam[2] - radius, 0.0, 0.0, 0.0]
    movel(target1, mod=DR_MV_MOD_REL, ref=DR_TOOL)

def pick_and_place(H_B_T_desire):
    '''
    Pick and Place process (mm)
    Pick: REL mode 
    Place: ABS mode
    '''
    # Picking process 
    tran_x,tran_y,tran_z = H_B_T_desire[:3,3]
    rot_mat = H_B_T_desire[:3,:3]
    rot_z1, rot_y, rot_z2 = rot_to_zyz(rot_mat)
    current_pose,_ = get_current_posx() 

    # Move to target object
    movel([tran_x,tran_y,current_pose[2],rot_z1, rot_y, rot_z2]) # move x,y direction and rotate
    movel([tran_x,tran_y,tran_z-5,rot_z1, rot_y, rot_z2]) # move z direction
    # movel([0,0,tran_z,0,0,0], mod=DR_MV_MOD_REL, ref=DR_TOOL)

    # Close gripper
    set_digital_output(1,1)
    time.sleep(0.5)

    # Move to targin bin
    movel([0.0, 0.0, -200, 0.0, 0.0, 0.0], mod=DR_MV_MOD_REL, ref=DR_TOOL)
    movel(TARGET_BIN)

    # Open gripper
    set_digital_output(1,0)
    time.sleep(0.5)

    # Come back home
    movel(HOMEX)



def rot_cam_y(radius: float, angle: float, action : int):
    '''
    Rotate camera 30 degree in y axis
    '''
    alpha = radians(angle)

    H_C1_C2_Y = np.array([[cos(alpha), 0, sin(alpha), -radius*sin(alpha)],
                        [0, 1, 0, 0],
                        [-sin(alpha), 0, cos(alpha), radius-radius*cos(alpha)],
                        [0, 0, 0, 1]]) ## rotate camera around y-axis wrt to camera (angle)

    ## Transformation of tool frames when rotate around y_axis of camera frame
    H_T1_T2_Y = np.matmul(H_T_C, np.matmul(H_C1_C2_Y, H_C_T)) 
    H_T2_T1_Y = inv(H_T1_T2_Y)

    if action == 0:
        theta1, theta2, theta3 = rot_to_zyz(H_T1_T2_Y)
        # print(theta1, theta2, theta3)
        # delta1 = [H_T1_T2_Y[0][3], H_T1_T2_Y[1][3], H_T1_T2_Y[2][3], theta1,angle,theta3]
        delta1 = [H_T1_T2_Y[0][3], H_T1_T2_Y[1][3], H_T1_T2_Y[2][3], theta1, theta2, theta3]
        movel(delta1, mod=DR_MV_MOD_REL, ref=DR_TOOL)
    
    if action == 1:
        theta1, theta2, theta3 = rot_to_zyz(H_T2_T1_Y)
        # print(theta1, theta2, theta3)
        # delta2 = [H_T2_T1_Y[0][3], H_T2_T1_Y[1][3], H_T2_T1_Y[2][3], theta1,-angle,theta3]
        delta2 = [H_T2_T1_Y[0][3], H_T2_T1_Y[1][3], H_T2_T1_Y[2][3], theta1, theta2, theta3]
        movel(delta2, mod=DR_MV_MOD_REL, ref=DR_TOOL)

def rot_cam_x(radius: float, angle: float, action : int):
    '''
    Rotate camera 30 degree in x axis
    '''
    alpha = radians(angle)
    H_C1_C2_X = np.array([[1, 0, 0, 0],
                        [0, cos(alpha), -sin(alpha), radius*sin(alpha)],
                        [0, sin(alpha), cos(alpha), radius - radius*cos(alpha)],
                        [0, 0, 0, 1]]) ## rotate camera around x-axis wrt to camera (-angle)

    ## Transformation of tool frame when rotate around x_axis of camera frame
    H_T1_T2_X = np.matmul(H_T_C, np.matmul(H_C1_C2_X, H_C_T))
    H_T2_T1_X = inv(H_T1_T2_X)

    if action == 0:
        theta1, theta2, theta3 = rot_to_zyz(H_T1_T2_X)
        # print(theta1,theta2,theta3)
        # delta1 = [H_T1_T2_X[0][3], H_T1_T2_X[1][3], H_T1_T2_X[2][3], theta1, angle, theta3]
        delta1 = [H_T1_T2_X[0][3], H_T1_T2_X[1][3], H_T1_T2_X[2][3], theta1, theta2, theta3]
        movel(delta1, mod=DR_MV_MOD_REL, ref=DR_TOOL)

    if action == 1:
        theta1, theta2, theta3 = rot_to_zyz(H_T2_T1_X)
        # print(theta1,theta2,theta3)
        # delta2 = [H_T2_T1_X[0][3], H_T2_T1_X[1][3], H_T2_T1_X[2][3], theta1, angle, theta3] 
        delta2 = [H_T2_T1_X[0][3], H_T2_T1_X[1][3], H_T2_T1_X[2][3], theta1, theta2, theta3] 
        movel(delta2, mod=DR_MV_MOD_REL, ref=DR_TOOL)





