# Autonomous Multi-view Bin-picking System 
This is comprehensive platform for autonomous bin-picking system. The system used multi-view scan-matching algorithm for acurate 6D pose estimation
[![demo](https://drive.google.com/file/d/1XvkiXS_cX4Qnfp5fu9LuZ5UZiqGllV_1/view?usp=drive_link)](https://drive.google.com/file/d/1XvkiXS_cX4Qnfp5fu9LuZ5UZiqGllV_1/view?usp=drive_link)
### Run full bin-picking system 
-__Step1__: Start the robot first. Then run the platform file.
```bash
roslaunch dsr_launcher single_robot_rviz.launch mode:=real host:=192.168.2.20
```
-__Step2__: Run the platform
```bash
rosrun auto_bin_picking bin_picking_platform.py
```
-__Step3__: Arrange object into bin. Make sure bin is aligned inside given ROI.
-__Step4__: On the color image, press "r" to start the picking process

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
