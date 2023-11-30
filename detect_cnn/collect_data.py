#!/usr/bin/env python3
import rospy
import cv2 
import pyrealsense2 as rs
import os,sys

sys.dont_write_bytecode = True
path = os.path.dirname(__file__)
print(path)
parent = os.path.abspath(os.path.join(path,"../"))
robot_setup_path = os.path.abspath(os.path.join(parent,"./src"))
sys.path.append(parent)
sys.path.append(robot_setup_path)

from Robot_Setup import *

# Home of robot 
HOMEX = [550, 70, 600, 0.0, 180.0, 0.0]

# Count to drop some first framse 
count = 0
count_img = 0

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

## Load the corner of bin 
X_CONNER = []
Y_CONNER = []
file = open(path + '/bin_corner.txt', 'r')
lines = file.readlines()
for line in lines:
    m = line.split(' ')
    X_CONNER.append(int(m[0]))
    Y_CONNER.append(int(m[1]))
file.close()

# Shutdown function 
def shutdown():
    global file
    print("shutdown time!")
    print("--------------")
    pub_stop.publish(stop_mode=STOP_TYPE_QUICK)
    pipeline.stop()
    return

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
    
    while not rospy.is_shutdown() :
        
        count += 1
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame or count < 10:
            continue

        # Convert color frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR) # convert RGB to BGR
        # Draw corner of bin 
        cv2.rectangle(color_image, (X_CONNER[0], Y_CONNER[0]), (X_CONNER[3], Y_CONNER[3]), (0,255,0), 4)

        # Convert depth frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data()) 
        depth_image_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.2), cv2.COLORMAP_JET) # convert to depth map

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Depth', depth_image_color)

        # Close 
        key = cv2.waitKey(100) & 0xFF
        
        if chr(key) == 'h':
            movel(HOMEX)

        if chr(key) == 'r':
            bin = color_image[Y_CONNER[0]:Y_CONNER[3], X_CONNER[0]:X_CONNER[3]]
            cv2.imwrite(path + "/valid_data/bin_" + str(count_img) + ".jpg", bin)
            count_img += 1

        if key == 27:
            cv2.destroyAllWindows()
            break

    pipeline.stop()