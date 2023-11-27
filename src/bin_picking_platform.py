#!/usr/bin/env python3
import rospy
import cv2 as cv
import pyrealsense2 as rs
import open3d as o3d
import os, sys
import copy

sys.dont_write_bytecode = True
dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(dir,"../")))

from move_robot_service import *
from detect_cnn.detect import *
from RegistrationPC import *

# from verify_point.RegistrationPC import combine_matching
np.set_printoptions(precision=3, suppress=True)

## Load the corner of bin 
X_CONNER = []
Y_CONNER = []

def load_bin_corner():
    path = os.path.abspath(os.path.join(dir,"../result"))
    file = open(path + '/bin_corner.txt', 'r')
    lines = file.readlines()
    for line in lines:
        m = line.split(' ')
        X_CONNER.append(int(m[0]))
        Y_CONNER.append(int(m[1]))
    file.close()

## Declare main class
class Robot_Scan():
    def __init__(self):
        
        # ROS setup
        rospy.init_node("robot_san_matching")
        rospy.on_shutdown(self.shutdown)
        self.pub_stop            = rospy.Publisher('/'+ROBOT_ID +ROBOT_MODEL+'/stop', RobotStop, queue_size=10)
        self.tf_sub              = rospy.Subscriber('/tf', TFMessage, self.callback_TF)
        self.tf_buffer           = tf2_ros.Buffer()
        self.tf_listener         = tf2_ros.TransformListener(self.tf_buffer)

        # Setup pointcloud 
        self.pc_record = o3d.geometry.PointCloud()

        # Setup camera 
        self.width = 1280 #1280
        self.height = 720 # 720
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, 30)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        self.frame = None
        self.color_frame = None
        self.depth_frame = None
        self.depth_frame_filter = None
        self.align = rs.align(rs.stream.color)

        self.depth_intrinsics = None
        
        # Start streaming
        self.pipeline.start(config)

        # image 
        self.color_image = None
        self.depth_image = None 
        self.depth_image_color = None
        self.bin_image = None
        self.detect_img = None
        self.color_roi = None
        self.target_object_center = None
        self.reset = True

        # Setup hole filling filter 
        self.hole_filling_filter = rs.hole_filling_filter(0)

        # Scan setup 
        self.radius = 300 # scan radius (mm)
        self.rotate_angle = 40 # scan angle (degree)
        self.angle_x = [self.rotate_angle]            
        self.angle_y = [self.rotate_angle]
        self.ROI = 150 # size of ROI (pixel * pixel)

        # Main loop
        time.sleep(0.5)
        self.main_loop()

    def main_loop(self):
        
        while not rospy.is_shutdown() :
            # Record frame
            self.record_frame()

            # Detect object 
            self.detect_img, self.target_object_center = detect_result(self.bin_color_image, self.bin_depth_image, model)
            
            # Show images
            cv.namedWindow('Depth', cv.WINDOW_AUTOSIZE)
            cv.imshow('Depth', self.depth_image_color)
            cv.namedWindow('Color', cv.WINDOW_AUTOSIZE)
            cv.imshow('Color', self.detect_img)

            # Click on color image
            # cv.setMouseCallback('Color', self.click_handle)

            # Process
            self.picking_process()

            # Shutdown 
            key = cv.waitKey(30) & 0xFF
            
            # Shutdown 
            if key == 27:
                cv.destroyAllWindows()
                self.pipeline.stop()
                break
            
            # Reset the bin
            if chr(key) == 'r': 
                self.reset = False

        # Shutdown 
        rospy.signal_shutdown("Shutting down ROS")
    
    def shutdown(self):
        ''' 
        Shutdown function
        '''     
        print("shutdown!")
        ## After 5 simulation, caculate average error  
        self.pub_stop.publish(stop_mode=STOP_TYPE_QUICK)
        return    

    def record_frame(self, update_aruco: bool = True):
        # Take frame
        self.frames = self.pipeline.wait_for_frames()
        self.frames = self.align.process(self.frames)
        
        self.color_frame = self.frames.first(rs.stream.color)
        self.depth_frame = self.frames.get_depth_frame()
        
        # Record color frame 
        self.color_image = np.asanyarray(self.color_frame.get_data())
        self.color_image = cv.cvtColor(self.color_image, cv.COLOR_RGB2BGR)
    
        # Record depth frame 
        self.depth_frame = rs.depth_frame(self.hole_filling_filter.process(self.depth_frame)) # apply hole filter     
        self.depth_image = np.asanyarray(self.depth_frame.get_data())
        self.depth_image_color = cv.applyColorMap(cv.convertScaleAbs(self.depth_image, alpha=0.2), cv.COLORMAP_JET) # convert to depth map
        
        # Draw corner of bin 
        self.bin_color_image = copy.deepcopy(self.color_image[Y_CONNER[0]:Y_CONNER[3], X_CONNER[0]:X_CONNER[3]])
        self.bin_depth_image = copy.deepcopy(self.depth_image[Y_CONNER[0]:Y_CONNER[3], X_CONNER[0]:X_CONNER[3]])
        cv.rectangle(self.color_image, (X_CONNER[0], Y_CONNER[0]),
                     (X_CONNER[3], Y_CONNER[3]), (0,255,0), 4)
        
        # Get the intrinsics of the color and depth cameras
        self.depth_intrinsics = self.depth_frame.profile.as_video_stream_profile().intrinsics    

    def record_point_cloud(self, ROI: int):

        x_mid = int(self.width/2) ; y_mid = int(self.height/2)
        # get center distance 
        depth_value_center = self.depth_frame.get_distance(x_mid, y_mid) 
        location_center = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [x_mid, y_mid], depth_value_center) # meter

        self.pc_record.clear() # clear point cloud

        # Take the color and depth ROI
        roi_x = int(x_mid - ROI/2)
        roi_y = int(y_mid - ROI/2)
        self.color_roi = self.color_image[roi_y:roi_y + ROI, roi_x:roi_x + ROI] # normalize [0,1]
        color_roi = cv.cvtColor(self.color_roi, cv.COLOR_RGB2BGR)
        depth_roi = self.depth_image[roi_y:roi_y + ROI, roi_x:roi_x + ROI]
        
        # Convert ROI to pc 
        image_pc = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color = o3d.geometry.Image(color_roi),
            depth = o3d.geometry.Image(depth_roi.astype(np.float32)),
            convert_rgb_to_intensity = False
        )

        self.pc_record = o3d.geometry.PointCloud.create_from_rgbd_image(
            image_pc,
            o3d.camera.PinholeCameraIntrinsic(
                self.ROI, self.ROI, 
                self.depth_intrinsics.fx, 
                self.depth_intrinsics.fy, 
                self.depth_intrinsics.ppx-roi_x, 
                self.depth_intrinsics.ppy-roi_y),
        )

        # Minus the radius 
        self.pc_record.translate([0,0,-self.radius/1000])
    
        rospy.loginfo("Record_PC")
        pass

    def save_data(self, index, update_aruco: bool = True):     
        path_pc    = os.path.abspath(os.path.join(os.path.dirname(__file__),"../data_pcl"))
        path_img = os.path.abspath(os.path.join(os.path.dirname(__file__),"../image")) 
        
        # Recordata
        for i in range(10):
            self.record_frame(update_aruco)

        # Save point cloud
        self.record_point_cloud(ROI = self.ROI)
        o3d.io.write_point_cloud(path_pc + "/pcl_view_" + str(index) + ".pcd", self.pc_record, True) 

        # Save image 
        cv.imwrite(path_img + "/image_view_" + str(index) + ".jpg", self.color_roi)
        cv.imwrite(path_img + "/image_depth_" + str(index) + ".png", self.depth_image_color)

    def click_handle(self,event, x_click, y_click, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.reset = False
            
    def picking_process(self):
        # Check if there is no object inside bin or or bin is reseted 
        if self.target_object_center == None or self.reset:
            self.reset = True
            return

        # Pixel location of target object
        x = X_CONNER[0] + self.target_object_center[0]
        y = Y_CONNER[0] + self.target_object_center[1]

        depth_value = self.depth_frame.get_distance(x, y) 
        location = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [x, y], depth_value) # meter
        obj_location = [m*1000 for m in location] # milimeter 

        # return if the measurement fails 
        if obj_location == [0., 0., 0.] or (not obj_location): 
            rospy.loginfo("Cannot detect object")
            return

        # Step0: Move robot right above click point on object
        move_to_scan_position(point=obj_location, radius=self.radius)
        
        # Step1: scan in the top view and Update the TF from base to scene 
        self.save_data(0, update_aruco=False) # Record point cloud from top view 
        H_C_S = np.eye(4)
        H_C_S[:3,3] = [0, 0, self.radius]
        H_B_C = self.get_tf(base_frame, camera_frame)
        global H_B_S, H_B_O
        H_B_S = np.matmul(H_B_C,H_C_S) 

        # Step2: Scan in x direction 
        if self.target_object_center[0] < int((X_CONNER[3]-X_CONNER[0])/2):
            self.angle_y = [-self.rotate_angle]
        else:
            self.angle_y = [self.rotate_angle]
        
        if self.target_object_center[1] > int((Y_CONNER[3]-Y_CONNER[0])/2):
            self.angle_x = [-self.rotate_angle]
        else: 
            self.angle_x = [self.rotate_angle]

        for i in range(len(self.angle_x)):
            rot_cam_x(self.radius, self.angle_x[i], 0)
            self.save_data("x_" + str(i), update_aruco=False)
            rot_cam_x(self.radius, self.angle_x[i], 1)

        # Step3: Scan in y direction 
        for i in range(len(self.angle_y)):
            rot_cam_y(self.radius, self.angle_y[i], 0)
            self.save_data("y_" + str(i), update_aruco=False)
            rot_cam_y(self.radius, self.angle_y[i], 1)

        # Step4: Pose estimation and publish the TF from base to object
        theta1, theta2, theta3, x_trans, y_trans, z_trans, rot_mat = combine_matching(rotate_angle_x = self.angle_x, rotate_angle_y = self.angle_y)
        H_S_O[:3,3] = [x_trans, y_trans, z_trans]
        H_S_O[:3,:3] = rot_mat
        H_B_O = np.matmul(H_B_S, H_S_O)
        public_tf("normal", [H_B_O], [base_frame], [target_object_frame])

        # # Skip if theta2 is too large:
        # print(theta2)
        # if abs(theta2) > 30:
        #     return

        # Step5: Pick and place
        H_B_T_desire = np.matmul(H_B_O, H_E_T)
        public_tf("normal", [H_B_T_desire], [base_frame], ["desire_tool_pose"])
        H_T_T_desire = self.get_tf(tool_frame, "desire_tool_pose") # TF from current tool to desired tool pose
        pick_and_place(H_B_T_desire)

    def callback_TF(self, msg: TFMessage):
        # Publish static transformation
        public_tf("normal", [H_T_C, H_T_E, H_B_S], 
                            [tool_frame, tool_frame, base_frame], 
                            [camera_frame, end_tool_frame, scene_frame])


    def get_tf(self, frame1: str, frame2: str):
        trans = self.tf_buffer.lookup_transform(frame1,frame2,rospy.Time(),rospy.Duration(0.5))
        matrix = np.eye(4)
        matrix[:3,3] = [trans.transform.translation.x*1000,
                       trans.transform.translation.y*1000,
                       trans.transform.translation.z*1000] # m to mm
        quat = [trans.transform.rotation.x, 
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w]
        rotation_matrix = transformations.quaternion_matrix(quat)
        matrix[:3,:3] = rotation_matrix[:3,:3]

        return matrix

## Start ##
def main():
    load_bin_corner() # setup conner for bin
    robot = Robot_Scan() # initialize platform
    rospy.spin()

if __name__ == "__main__":
    main()