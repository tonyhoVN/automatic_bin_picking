# Multi Views Pose Estimation 
This is multi-scan algorithm for exact pose estimation

### Run System with algorithm 
Start the robot first. Then run the algorithm.
```bash
roslaunch dsr_launcher single_robot_rviz.launch mode:=real host:=192.168.2.20
rosrun auto_bin_picking bin_picking_platform.py
```

### Calibration hand-eye camera 
Execute following commands to setup robot 
```bash
roslaunch dsr_launcher dsr_moveit.launch mode:=real host:=192.168.2.20
rosservice call /dsr01m1013/system/set_robot_mode 0
```
In another terminal, open camera 
```bash
roslaunch realsense2_camera rs_camera.launch align_depth:=true
```

### Setting camera
Use realsense-viewer to change the camera exposure
```bash
roslaunch realsense2_camera rs_camera.launch initial_reset:=true 
realsense-viewer
```

### Change network to use wifi
```bash
route -n
sudo ifmetric wlx588694f57751 50
```
