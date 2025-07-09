cmake_minimum_required(VERSION 3.16)
project(grid_map_filter_node)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(grid_map_ros REQUIRED)
find_package(grid_map_filters REQUIRED)
find_package(median_fill_filter REQUIRED)
find_package(pluginlib REQUIRED)
find_package(grid_map_msgs REQUIRED)

include_directories(
  include 
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

add_executable(filters_demo_node
  src/filters_demo_node.cpp
  src/FiltersDemo.cpp
)

ament_target_dependencies(filters_demo_node
  rclcpp
  grid_map_ros
  grid_map_filters
  median_fill_filter
  pluginlib
  grid_map_msgs
)

install(TARGETS filters_demo_node 
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

