cmake_minimum_required(VERSION 3.0.2)
project(multi_scan)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find Packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  message_generation
)

## Add service 
add_service_files(
  FILES
  RecordPcl.srv
  MergePcl.srv
  MoveMultipleView.srv
  RotateMultipleView.srv
  GoHome.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)


## Setup catkin_make
catkin_package(
  CATKIN_DEPENDS message_runtime 
  LIBRARIES ${PROJECT_NAME}
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)


