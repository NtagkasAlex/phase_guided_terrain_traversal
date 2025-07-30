#!/bin/bash

# Source your ROS 2 setup


# Run transform_sensors node
#cd ~/autonomy_stack_go2
#source install/setup.bash
#ros2 run transform_sensors transform_everything &
#TRANSFORM_PID=$!

# Run point_lio launch file


# Run your Python script

cd ~/Desktop
python tf_publish.py&
PY_PID=$!

cd ~/elevation_2
colcon build 
source install/setup.bash
# Wait for all processes to finish
ros2 launch elevation_mapping_demos ground_truth_demo.launch.xml &
ELE_PID=$!
wait $PY_PID
wait $ELE_PID

