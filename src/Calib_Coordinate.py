#!/usr/bin/env python3
import rospy
import cv2 
import pyrealsense2 as rs
import os,sys

sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from Scan_init_single_cam import *

# Count to drop some first framse 
count = 0

# U,V coordinate of selected point in color image
U_ = 0; V_ = 0
FLAG = 0 # flag for mouse click

# Mouse click function 
def get_pixel_value(event, x, y, flags, param):
    global U_, V_, FLAG
    if event == cv2.EVENT_LBUTTONDOWN:
        U_ = x
        V_ = y
        FLAG = flags

# Shutdown function 
def shutdown():
    global file
    print("shutdown time!")
    print("--------------")
    pub_stop.publish(stop_mode=STOP_TYPE_QUICK)
    pipeline.stop()
    return

# Calib 
E1_B = [540.3847045898438, 40.74516296386719, 500.8709716796875]
O_C = [0, 0, 0]
E2_B = [0, 0, 0]
file = open(os.path.dirname(__file__) + "/object_position.txt","w")
X_C_T = []
Y_C_T = []

def record_calib_data():
    global E2_B, O_C, E1_B, X_C_T, Y_C_T
    current_pos,_ = get_current_posx()
    E2_B = current_pos[:3]
    E2_E1 = np.array(E2_B) - np.array(E1_B)
    C_B = [O_C[0] + E2_E1[0], O_C[1] - E2_E1[1]]
    # print(C_B)

    X_C_T.append(C_B[0])
    Y_C_T.append(C_B[1])
    file.write(str(C_B[0]) + " " + str(C_B[1]) + " \n")
    calib_calculation()


def calib_calculation():
    # data = np.loadtxt("output.txt", dtype=float)
    # print(data)
    # x = data[:,0]; y = data[:,1]
    # x_avg = np.average(x); y_avg = np.average(y)
    # print(x_avg, y_avg)

    global X_C_T, Y_C_T
    x_avg = np.average(np.array(X_C_T))
    y_avg = np.average(np.array(Y_C_T))
    print(x_avg, y_avg)




### MAIN ####
if __name__ == "__main__":

    # Setup ROS
    rospy.init_node('dsr_simple_test_py')
    rospy.on_shutdown(shutdown)
    pub_stop = rospy.Publisher('/'+ROBOT_ID +ROBOT_MODEL+'/stop', RobotStop, queue_size=10)
    
    ## Setup robot before running
    movel(HOMEX) # move robot to home postition first time 
    wait(1)

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    # Start streaming
    pipeline.start(config)

    # intrinsic 
    profile = pipeline.get_active_profile()
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    color_intrinsics = color_profile.get_intrinsics()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    print(np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx], [0, color_intrinsics.fy, color_intrinsics.ppy], [0, 0, 1]]))
    print([color_intrinsics.coeffs[0], color_intrinsics.coeffs[1], color_intrinsics.coeffs[2], color_intrinsics.coeffs[3], color_intrinsics.coeffs[4]])

    # displacement 
    increment = 1 #mm
    
    while not rospy.is_shutdown() :

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame or count < 10:
            count += 1
            continue

        # # Get the intrinsics of the color and depth cameras
        # depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        # color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        # # Apply hole filling filter for depth frame 
        # hole_filling = rs.hole_filling_filter()
        # filter_depth_frame = hole_filling.process(depth_frame) # convert from depth_frame to disparity and filter 
        # depth_frame = rs.depth_frame(filter_depth_frame) # convert from disparity back to depth_frame

        # Convert color frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR) # convert RGB to BGR
        # cv2.circle(color_image, (320,240), 5, (255,0,0),2) 
        # cv2.line(color_image,(320,240),(320,250),(0,255,0),1)
        # cv2.line(color_image,(320,240),(330,240),(0,255,0),1)

        # Convert depth frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data()) 
        depth_image_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.2), cv2.COLORMAP_JET) # convert to depth map

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Depth', depth_image_color)

        # Get specific pixel (u,v) and its depth value        
        cv2.setMouseCallback('RealSense', get_pixel_value)

        if FLAG == 1:
            
            u = U_; v = V_

            # Get the depth value of (u,v)
            depth_value = depth_frame.get_distance(u, v)

            # Map the pixel (u, v) to the 3D point (x, y, z) in depth cam coordinate 
            point_cam = rs.rs2_deproject_pixel_to_point(color_intrinsics, [u, v], depth_value) # meter
            point_cam_mm = [point*1000 for point in point_cam]
            
            # Record position of object
            O_C = point_cam_mm
            for point in point_cam_mm:
                print(format(point,".3f"))
            print("\n")

            # Record position of tool 
            current_pos,_ = get_current_posx()
            E1_B = current_pos[:3]


            # Map point in deptp cam coordinate to tool-flage coordinate 
            point_tool = [point_cam_mm[0] + H_T_C[0][3],
                          point_cam_mm[1] + H_T_C[1][3],
                          point_cam_mm[2] + H_T_C[2][3]] # mm
            
            # Move robot to center of object
            movel([point_tool[0], point_tool[1], point_tool[2] - Z_T_E-2, 0, 0, 0], mod=DR_MV_MOD_REL, ref=DR_TOOL)

            # # own problem 
            # x = (u - color_intrinsics.ppx) / color_intrinsics.fx *depth_value
            # y = (v - color_intrinsics.ppy) / color_intrinsics.fy *depth_value
            # print(x, y, depth_value)
            
        #     # Move robot to targeted point (relative to tool frame)
        #     movel([H_T_C[0][3], H_T_C[1][3], 235 + H_T_C[2][3]-80, 0, 0, 0], mod=DR_MV_MOD_REL, ref=DR_TOOL)
            # time.sleep(5)
            # rot_cam_x(radius=100, angle=20, action=0)
            # time.sleep(5)
            # rot_cam_x(radius=0, angle=20, action=1)

        # if FLAG == 0:
            
        #     # # Map the pixel (U, V) in color image to (u,v) pixel in depth image 
        #     # u = int(U_*scale_x + x0)
        #     # v = int(V_*scale_y + y0)
        #     u = 320; v = 240

        #     # Get the depth value of (u,v)
        #     depth_value = depth_frame.get_distance(u, v)
            

        #     if depth_value == [0.0,0.0,0.0]: continue

        #     # Map the pixel (u, v) to the 3D point (x, y, z) in depth cam coordinate 
        #     point_cam = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth_value) # meter
        #     point_cam = [point*1000 for point in point_cam]

        #     print(point_cam)

        # Reset flag
        FLAG = 0  

        # Close 
        key = cv2.waitKey(100) & 0xFF
        
        if chr(key) == 'h': #q
            movel(HOMEX)

        if chr(key) == 'e': #e

                        # # Map the pixel (U, V) in color image to (u,v) pixel in depth image 
            # u = int(U_*scale_x + x0)
            # v = int(V_*scale_y + y0)
            u = 320; v = 240

            # Get the depth value of (u,v)
            depth_value = depth_frame.get_distance(u, v)
            print(depth_value)

            if depth_value == [0.0,0.0,0.0]: continue

            # Map the pixel (u, v) to the 3D point (x, y, z) in depth cam coordinate 
            point_cam = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth_value) # meter
            point_cam = [point*1000 for point in point_cam]

            # # Map point in deptp cam coordinate to tool-flage coordinate 
            # point_tool = [point_cam[0] + H_T_C[0][3],
            #               point_cam[1] + H_T_C[1][3],
            #               point_cam[2] + H_T_C[2][3]] # mm
            
            for point in point_cam:
                print(format(point,".3f"))
            
            
            # Move robot to targeted point (relative to tool frame)
            movel([0 + H_T_C[0][3], 0 + H_T_C[1][3], 238 + H_T_C[2][3]-81, 0, 0, 0], mod=DR_MV_MOD_REL, ref=DR_TOOL)

        if chr(key) == 'r': #r: 
            ''' 
            Press r to capture the (320, 240)
            '''

            u = 320; v = 240
            depth_value = depth_frame.get_distance(u, v)
            if depth_value == 0: 
                print("Not measure")
                continue
            point = rs.rs2_deproject_pixel_to_point(color_intrinsics, [u, v], depth_value) # meter
            
            x = point[0]/point[2]*color_intrinsics.fx
            y = point[1]/point[2]*color_intrinsics.fy

            print(point)
            print(x,y,point[2])

        # JOG
        if chr(key) == 'a':
            movel([-increment,0,0,0,0,0], mod = DR_MV_MOD_REL, ref=DR_TOOL)
        if chr(key) == 'd':
            movel([increment,0,0,0,0,0], mod = DR_MV_MOD_REL, ref=DR_TOOL)
        if chr(key) == 'w':
            movel([0,increment,0,0,0,0], mod = DR_MV_MOD_REL, ref=DR_TOOL)  
        if chr(key) == 's':
            movel([0,-increment,0,0,0,0], mod = DR_MV_MOD_REL, ref=DR_TOOL)
        if chr(key) == 'o':
            movel([0,0,increment,0,0,0], mod = DR_MV_MOD_REL, ref=DR_TOOL)
        if chr(key) == 'l':
            movel([0,0,-increment,0,0,0], mod = DR_MV_MOD_REL, ref=DR_TOOL)

        # Calib process:
        if chr(key) == 'c':
            record_calib_data() 
        
        if chr(key) == 'f':
            current,_ = get_current_posx()
            print([current[0], current[1], current[2]])
            file.writelines('\t'.join(map(str, [current[0], current[1], current[2]])))

        
        # Incremental
        if chr(key) == '1':
            increment = 10
        if chr(key) == '2':
            increment = 1
        if chr(key) == '3':
            increment = 0.5
        if chr(key) == '4':
            increment = 0.1

        if key == 27: #esc
            cv2.destroyAllWindows()
            break
    
    file.close()
    pipeline.stop()