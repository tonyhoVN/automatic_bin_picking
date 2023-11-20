#!/usr/bin/env python3
import rospy
import cv2 as cv
import pyrealsense2 as rs
import open3d as o3d
import random

from bin_picking.srv import *
from RegistrationPC import *
from Robot_Setup import *

# from verify_point.RegistrationPC import combine_matching
np.set_printoptions(precision=3, suppress=True)

sys.dont_write_bytecode = True
dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(dir))
path = os.path.abspath(os.path.join(dir,"../result"))
sys.path.append(path)

## Load the corner of bin 
X_CONNER = []
Y_CONNER = []

def load_bin_corner():
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
        self.move_robot_client   = rospy.ServiceProxy("move_robot_service", MoveMultipleView)
        self.rotate_robot_client = rospy.ServiceProxy("rotate_camera_service", RotateMultipleView)

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

        # Setup hole filling filter 
        self.hole_filling_filter = rs.hole_filling_filter(0)

        # Scan setup 
        self.radius = 300 # scan radius (mm)
        self.rotate_angle = 40 # scan angle (degree)
        self.angle_x = [self.rotate_angle]            
        self.angle_y = [self.rotate_angle]
        self.ROI = 300 # size of ROI (pixel * pixel)

        # Main loop
        self.main_loop()

    def main_loop(self):
        
        while not rospy.is_shutdown() :
            # Record frame
            self.record_frame()
            
            # Show images
            cv.namedWindow('Color', cv.WINDOW_AUTOSIZE)
            cv.imshow('Color', self.color_image)
            cv.namedWindow('Depth', cv.WINDOW_AUTOSIZE)
            cv.imshow('Depth', self.depth_image_color)

            # Click on color image
            cv.setMouseCallback('Color', self.click_handle)

            # Shutdown 
            key = cv.waitKey(30) & 0xFF
            if key == 27:
                cv.destroyAllWindows()
                self.pipeline.stop()

                break

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
        
        # Draw corner of bin 
        cv.rectangle(self.color_image, (X_CONNER[0], Y_CONNER[0]),
                     (X_CONNER[3], Y_CONNER[3]), (0,255,0), 4)

        # Record depth frame 
        self.depth_frame = rs.depth_frame(self.hole_filling_filter.process(self.depth_frame)) # apply hole filter     
        self.depth_image = np.asanyarray(self.depth_frame.get_data())
        self.depth_image_color = cv.applyColorMap(cv.convertScaleAbs(self.depth_image, alpha=0.2), cv.COLORMAP_JET) # convert to depth map
        
        # Get the intrinsics of the color and depth cameras
        self.depth_intrinsics = self.depth_frame.profile.as_video_stream_profile().intrinsics    

    def click_handle(self,event, x_click, y_click, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            
            # Simulation of 5 clicks with a deviation of 10 pixels  
        
            x = x_click 
            y = y_click 
            # x = x_click 
            # y = y_click 
            # Get the position of object 
            depth_value = self.depth_frame.get_distance(x, y) 
            location = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [x, y], depth_value) # meter
            obj_location = [m*1000 for m in location] # milimeter 

            # return if the measurement fails 
            if obj_location == [0., 0., 0.] or (not obj_location): 
                rospy.loginfo("Cannot detect object")
                return

            # Step0: Move robot right above click point on object
            if rospy.wait_for_service("move_robot_service",2): return
            move_client = MoveMultipleViewRequest(point = obj_location, radius = self.radius)
            move_client.action = 0
            self.move_robot_client(move_client)
            
            # Step1: scan in the top view and Update the TF from base to scene 
            self.save_data(0, update_aruco=False) # Record point cloud from top view 
            H_C_S = np.eye(4)
            H_C_S[:3,3] = [0, 0, self.radius]
            H_B_C = self.get_tf(base_frame, camera_frame)
            global H_B_S, H_B_O
            H_B_S = np.matmul(H_B_C,H_C_S) 

            # Step2: scan around x axis 
            for i in range(len(self.angle_x)):
                if rospy.wait_for_service("rotate_camera_service", 2): return
                self.rotate_robot_client(axis = 0, radius = self.radius, angle = self.angle_x[i], action = 0)
                self.save_data("x_" + str(i), update_aruco=False)
                self.rotate_robot_client(axis = 0, radius = self.radius, angle = self.angle_x[i], action = 1)

            # Step3: scan around y axis 
            for i in range(len(self.angle_y)):
                if rospy.wait_for_service("rotate_camera_service", 2): return
                self.rotate_robot_client(axis = 1, radius = self.radius, angle = self.angle_y[i], action = 0)
                self.save_data("y_" + str(i), update_aruco=False)
                self.rotate_robot_client(axis = 1, radius = self.radius, angle = self.angle_y[i], action = 1)

            # Step4: Matching process --> pose estimation
        

    def record_point_cloud(self, ROI: int):

        x_mid = int(self.width/2) ; y_mid = int(self.height/2)
        # get center distance 
        depth_value_center = self.depth_frame.get_distance(x_mid, y_mid) 
        location_center = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [x_mid, y_mid], depth_value_center) # meter

        self.pc_record.clear() # clear point cloud

        # Take the color and depth ROI
        roi_x = int(x_mid - ROI/2)
        roi_y = int(y_mid - ROI/2)
        color_roi = self.color_image[roi_y:roi_y + ROI, roi_x:roi_x + ROI] # normalize [0,1]
        color_roi = cv.cvtColor(color_roi, cv.COLOR_RGB2BGR)
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
        cv.imwrite(path_img + "/image_view_" + str(index) + ".jpg", self.color_image)
        cv.imwrite(path_img + "/image_depth_" + str(index) + ".png", self.depth_image_color)


    def calculate_error(self, theta1, theta2, theta3, x_trans, y_trans, z_trans, error_mat):
        error_mat = self.get_tf(reference_frame, target_object_frame)
        # error_trans = sqrt(error_mat[0,3]**2 + error_mat[1,3]**2 + error_mat[3,3]**2)
        error_trans = sqrt(error_mat[0,3]**2 + error_mat[1,3]**2 + error_mat[2,3]**2)
        error_rot = degrees(np.arccos((np.trace(error_mat[:3,:3])-1)/2))

        if error_rot > 90:
            error_rot -= 180 

        print("Translation error: %.2f" %(error_trans))
        print("Rotation error: %.2f" %(error_rot))
        return error_trans, abs(error_rot)
    
    def calculate_average_error(self, error_tran, error_rot):
        error_tran_avg = np.average(np.array(error_tran))
        error_rot_avg = np.average(np.array(error_rot))
        print("Success detect: %d/10" % len(error_rot))
        print("Translation Error: %.2f mm" % error_tran_avg)
        print("Rotation Error: %.2f deg" % error_rot_avg)

    def callback_TF(self, msg: TFMessage):
        # Publish static transformation
        public_tf("normal", [H_T_C, H_T_E, H_B_S, H_B_R], 
                            [tool_frame, tool_frame, base_frame, base_frame], 
                            [camera_frame, end_tool_frame, scene_frame, reference_frame])

        # Find tranformation
        pass


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