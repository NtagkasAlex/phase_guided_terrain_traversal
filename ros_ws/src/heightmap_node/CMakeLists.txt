cmake_minimum_required(VERSION 3.5)
project(heightmap_node)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Required dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(grid_map_ros REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

# Define the executable
add_executable(heightmap_node src/heightmap_node.cpp)

# Link dependencies to the executable
ament_target_dependencies(heightmap_node
  rclcpp
  sensor_msgs
  grid_map_msgs
  grid_map_ros
  tf2
  tf2_geometry_msgs
  Eigen3
)

# Install the executable
install(TARGETS heightmap_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install the launch files
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# Package export
ament_package()

