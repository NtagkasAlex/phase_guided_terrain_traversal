#!/bin/bash

# Source your ROS 2 setup


# Run transform_sensors node
#cd ~/autonomy_stack_go2
#source install/setup.bash
#ros2 run transform_sensors transform_everything &
#TRANSFORM_PID=$!

# Run point_lio launch file

cd ~/catkin_point_lio_unilidar
colcon build --packages-select point_lio
source install/setup.bash
ros2 run transform_sensors transform_everything &
ros2 launch point_lio mapping_unilidar_l1.launch.py &
LIO_PID=$!

# Run your Python script

#python3 ~/Desktop/pcd.py &
#python3 ~/Desktop/static.py &
ros2 run tf2_ros static_transform_publisher 0 0 0.3 0 0 0 start map &
ros2 run tf2_ros static_transform_publisher -0.293 0 0.06 0 0 0 odom base_link &
PCD_PID=$!

# Wait for all processes to finish
wait $TRANSFORM_PID
wait $LIO_PID
wait $PCD_PID

