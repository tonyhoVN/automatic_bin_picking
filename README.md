# Multi Views Pose Estimation 
This is multi-scan algorithm for exact pose estimation

### Command for launch robot
```bash
roslaunch dsr_launcher single_robot_rviz.launch mode:=real host:=192.168.2.20
roslaunch dsr_launcher dsr_moveit.launch mode:=real host:=192.168.2.20
```
# Change network
```bash
route -n
sudo ifmetric wlx588694f57751 50
```

# publish pointcloud from camera 
rosrun bin_picking pcl_publish

# Run test control bin_picking
rosrun bin_picking move_robot_to_target


# maching multiple view of objects
pcl_matching_multiple 

# matching single view with 


# run full bin_picking C++
rosrun bin_picking move_robot_service 
rosrun bin_picking robot_scan
rosrun bin_picking pcl_matching_test1

# run full bin_picking py
rosrun bin_picking move_robot_service 
rosrun bin_picking robot_scan.py

# Camera calib
roslaunch dsr_control dsr_moveit_cali.launch 
rosservice call /dsr01m1013/system/set_robot_mode 0

# Setting camera
roslaunch realsense2_camera rs_camera.launch initial_reset:=true 
realsense-viewer
