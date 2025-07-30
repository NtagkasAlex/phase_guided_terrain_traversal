#!/bin/bash

# Source your ROS 2 setup


# Run transform_sensors node
#cd ~/autonomy_stack_go2
#source install/setup.bash
#ros2 run transform_sensors transform_everything &
#TRANSFORM_PID=$!

# Run point_lio launch file

cd ~/ros_scripts
./run_point_lio.sh&
LIO_PID=$!

# Run your Python script

cd ~/elevation
colcon build --packages-select elevation_mapping grid_map_filter_node --cmake-args -DCMAKE_BUILD_TYPE=Release 
source install/setup.bash
# Wait for all processes to finish
ros2 launch elevation_mapping elevationMapping_launch.py &
ros2 launch heightmap_node heightmap_launch.py  &
ros2 launch grid_map_filter_node filters_demo.launch.py  &
ELE_PID=$!
wait $LIO_PID
wait $ELE_PID

