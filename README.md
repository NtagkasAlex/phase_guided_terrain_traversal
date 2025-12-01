# PGGT
A perceptive reinforcement learning locomotion framework developed for the Unitree GO2 in simulation (MuJoco Playground) and deployed using unitree sdk2py, on real hardware.
## Perception pipeline using docker
### Setup
```bash
docker volume create ros2_ws_build
docker volume create ros2_ws_install
```
Build:
```bash
docker build -t ros2-humble-dev .
```
Run:
```bash
docker run -it --rm   -e DISPLAY=$DISPLAY   -e LIBGL_ALWAYS_SOFTWARE=1   -e MESA_LOADER_DRIVER_OVERRIDE=llvmpipe   -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v ros2_ws_build:/root/ros_ws/build   -v ros2_ws_install:/root/ros_ws/install   ros2-humble-dev 
```
For the first time:
```bash
source /opt/ros/humble/setup.bash
cd ros_ws
colcon build --merge-install --install-base /root/ros_ws/install --symlink-install
```
### Running the pipeline
Make sure /utlidar/cloud and /utlidar/imu topics are providing data.
```bash
docker run -it --rm   -e DISPLAY=$DISPLAY   -e LIBGL_ALWAYS_SOFTWARE=1   -e MESA_LOADER_DRIVER_OVERRIDE=llvmpipe   -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v ros2_ws_build:/root/ros_ws/build   -v ros2_ws_install:/root/ros_ws/install   ros2-humble-dev 
```
Inside container:
```bash
cd ros_scripts
chmod +x run_elevation.sh
./run_elevation.sh
```
An rviz window should showup, showing the gridmap and the scandots.
