#!/bin/bash

cd ~/Desktop/phase_guided_terrain_traversal/ros_ws
colcon build --packages-select point_lio
source install/setup.bash
ros2 run transform_sensors transform_everything &
ros2 launch point_lio mapping_unilidar_l1.launch.py &
LIO_PID=$!

ros2 run tf2_ros static_transform_publisher 0 0 0.3 0 0 0 start map &
ros2 run tf2_ros static_transform_publisher -0.293 0 0.06 0 0 0 odom base_link &
PCD_PID=$!

wait $TRANSFORM_PID
wait $LIO_PID
wait $PCD_PID

