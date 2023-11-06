#!/usr/bin/env python3
import rospy
from bin_picking.srv import *
from Scan_init_single_cam import *

# sys.dont_write_bytecode = True
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# print(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../doosan-robot/common/imp")) )

size_box = [30,20,40]


## Class 
class Move_Robot():
    def __init__(self):      
        # Setup ROS
        rospy.init_node('multi_scan_service')
        rospy.on_shutdown(self.shutdown)
        self.pub_stop           = rospy.Publisher('/'+ROBOT_ID +ROBOT_MODEL+'/stop', RobotStop, queue_size=10)
        self.pick_object_server = rospy.Service("move_robot_service", MoveMultipleView, self.move_close_object)
        self.rotate_server      = rospy.Service("rotate_camera_service", RotateMultipleView, self.rotate_camera) 
        self.go_home_server     = rospy.Service("go_home", GoHome, self.go_home)

        # Public static tf 
        # public_tf("static", H_T_E, tool, end_tool)

        ## Setup robot before running
        # movej(HOME) 
        movel(HOMEX) # move robot to home postition first time 
        wait(1)

    def shutdown(self):
        ''' 
        Shutdown function
        '''     
        print("shutdown!")
        print("--------------")
        self.pub_stop.publish(stop_mode=STOP_TYPE_QUICK)
        return    
    
    def move_close_object(self, req : MoveMultipleViewRequest):
        '''
        Service that moves robot close to object 
        '''
        ## REMEMBER: coordinate in milimeter ##

        point_object_cam = list(req.point) # distance from object to depth camera  

        
        # Calculate distance from object to tool
        point_object_tool = [0,0,0]
        point_object_tool[0] = point_object_cam[0] + H_T_C[0][3] 
        point_object_tool[1] = point_object_cam[1] + H_T_C[1][3]
        point_object_tool[2] = point_object_cam[2] + H_T_C[2][3]

        # target2 = [H_T_C[0][3], H_T_C[1][3], (2/3)*float(point_object_cam[2]) + H_T_C[2][3] - 50, 0, 0, 0]
        
        
        if req.action == 0:
            rospy.loginfo("Come to object, start scanning")
            # Move to position above box
            # target1 = [point_object_cam[0], point_object_cam[1], (1/3)*float(point_object_cam[2]), 0, 0, 0]
            print(point_object_cam[0], point_object_cam[1])
            target1 = [point_object_cam[0], point_object_cam[1], point_object_cam[2] - req.radius, 0.0, 0.0, 0.0]
            movel(target1, mod=DR_MV_MOD_REL, ref=DR_TOOL)

        if req.action == 1:
            rospy.loginfo("Picking object")
            
            # Come to center 
            target1 = [point_object_tool[0], point_object_tool[1], 0.0, 0.0, 0.0, 0.0]
            movel(target1, mod=DR_MV_MOD_REL, ref=DR_TOOL)

            # Rotate align to object 
            target2 = [0.0, 0.0, 0.0, req.angle[0], 0, req.angle[2]]
            movel(target2, mod=DR_MV_MOD_REL, ref=DR_TOOL)

            # Come close to center of object
            z_object_end_tool = point_object_tool[2] - Z_T_E - 1
            target3 = [0.0, 0.0, z_object_end_tool, 0.0, 0.0, 0.0]
            movel(target3, mod=DR_MV_MOD_REL, ref=DR_TOOL)
            time.sleep(1)
            
            '''
            # Close gripper
            set_digital_output(1,1)
            time.sleep(0.5)

            # Move to targin bin
            movel([0.0, 0.0, -z_object_end_tool, 0.0, 0.0, 0.0], mod=DR_MV_MOD_REL, ref=DR_TOOL)
            movel(TARGET_BIN)

            # Open gripper
            set_digital_output(1,0)

            # Move to Home after grasping
            movel(HOMEX)

            '''

        return MoveMultipleViewResponse()
    

    def rotate_camera(self, req : RotateMultipleViewRequest):
        '''
        Service that rotates the camera to scan in multiple views
        '''
        if req.axis == 0:
            rospy.loginfo("Scan in x axis")
            rot_cam_x(radius= req.radius, angle=req.angle, action=req.action)
        if req.axis == 1:
            rospy.loginfo("Scan in y axis")
            rot_cam_y(radius= req.radius, angle=req.angle, action=req.action)

        return RotateMultipleViewResponse()
    
    def go_home(self):
        movel(HOMEX)



### MAIN ####
if __name__ == "__main__":
    robot = Move_Robot()
    rospy.spin()



