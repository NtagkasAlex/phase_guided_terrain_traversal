cmake_minimum_required(VERSION 3.10)
project(median_fill_filter)

## Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(grid_map_core REQUIRED)
find_package(grid_map_cv REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pluginlib REQUIRED)
find_package(Eigen3 REQUIRED)

## Declare your include directories
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Library
add_library(${PROJECT_NAME}
  src/MedianFillFilter.cpp
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  grid_map_core
  grid_map_cv
  pluginlib
  Eigen3
  OpenCV
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
)


## Install
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(
  FILES plugin_description.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

