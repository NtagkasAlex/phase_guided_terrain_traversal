#!/bin/bash

cd ~/Desktop
python joint_pub.py&
python static.py&
PY_PID=$!

cd ~/robot_state_pub
colcon build 
source install/setup.bash
# Wait for all processes to finish
ros2 launch robot_state_publisher go2.py &
ELE_PID=$!
wait $PY_PID
wait $ELE_PID

