#!/bin/bash

./run_point_lio.sh&
LIO_PID=$!



cd ../ros_ws
colcon build --merge-install --packages-select elevation_mapping grid_map_filter_node #--cmake-args -DCMAKE_BUILD_TYPE=Release 
source install/setup.bash
# Wait for all processes to finish
ros2 launch elevation_mapping elevationMapping_launch.py &
ros2 launch heightmap_node heightmap_launch.py  &
ros2 launch grid_map_filter_node filters_demo.launch.py  &
ELE_PID=$!
wait $LIO_PID
wait $ELE_PID

