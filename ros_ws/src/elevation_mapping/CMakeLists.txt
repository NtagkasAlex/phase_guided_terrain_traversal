cmake_minimum_required(VERSION 3.5.1)
project(elevation_mapping)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)
# find_package(eigen_conversions REQUIRED) probably not necessary
find_package(grid_map_core REQUIRED)
find_package(grid_map_ros REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(kindr REQUIRED)
find_package(kindr_ros REQUIRED)
find_package(message_filters REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(Boost REQUIRED COMPONENTS system thread)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(Threads REQUIRED)



set(DEPENDENCIES
  grid_map_core
  grid_map_ros
  grid_map_msgs
  kindr
  kindr_ros
  message_filters
  pcl_ros
  rclcpp
  sensor_msgs
  std_msgs
  std_srvs
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  geometry_msgs
  # Threads
)

# catkin_package(
#   INCLUDE_DIRS
#     include
#     ${Eigen_INCLUDE_DIRS}
#   LIBRARIES
#     ${PROJECT_NAME}_library
#   CATKIN_DEPENDS
#     ${CATKIN_PACKAGE_DEPENDENCIES}
#   DEPENDS
#     Boost
# )

include_directories(
  include
  SYSTEM
    ${Boost_INCLUDE_DIRS}
    # ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

###########
# Library #
###########

# Our custom pcl type with precompiled template instantiations.
add_library(${PROJECT_NAME}_pcl_types
  src/PointXYZRGBConfidenceRatio.cpp
)

target_link_libraries(${PROJECT_NAME}_pcl_types
  ${PCL_LIBRARIES} 
)

ament_target_dependencies(${PROJECT_NAME}_pcl_types
  ${DEPENDENCIES}
)

add_library(${PROJECT_NAME}_library
  src/ElevationMapping.cpp
  src/ElevationMap.cpp
  src/input_sources/Input.cpp
  src/input_sources/InputSourceManager.cpp
  src/postprocessing/PostprocessorPool.cpp
  src/postprocessing/PostprocessingWorker.cpp
  src/postprocessing/PostprocessingPipelineFunctor.cpp
  src/RobotMotionMapUpdater.cpp
  src/sensor_processors/SensorProcessorBase.cpp
  src/sensor_processors/StructuredLightSensorProcessor.cpp
  src/sensor_processors/StereoSensorProcessor.cpp
  src/sensor_processors/LaserSensorProcessor.cpp
  src/sensor_processors/PerfectSensorProcessor.cpp
)

ament_target_dependencies(${PROJECT_NAME}_library
  ${DEPENDENCIES}
)

target_link_libraries(${PROJECT_NAME}_library
  ${PROJECT_NAME}_pcl_types
  ${Boost_LIBRARIES}
  pthread  
)

##############
# Executable #
##############
add_executable(get_grid_map_client src/test_service.cpp)

ament_target_dependencies(get_grid_map_client rclcpp grid_map_msgs)

install(TARGETS
  get_grid_map_client
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(${PROJECT_NAME}
  src/elevation_mapping_node.cpp
)

add_executable(listener
  src/test.cpp
)

ament_target_dependencies(${PROJECT_NAME}
  ${DEPENDENCIES}
)

ament_target_dependencies(listener
  ${DEPENDENCIES}
)


target_link_libraries(${PROJECT_NAME}
  ${PROJECT_NAME}_library
)

#############
## Install ##
#############

install(TARGETS
    # ${PROJECT_NAME}
  ${PROJECT_NAME}_pcl_types
  ${PROJECT_NAME}_library
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS
  ${PROJECT_NAME}
  listener
  DESTINATION lib/${PROJECT_NAME}
)

# install(
#   DIRECTORY 
#     include/${PROJECT_NAME}/
#   DESTINATION 
#     ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
# )

install(DIRECTORY config launch sample_data rviz2
  DESTINATION share/${PROJECT_NAME}
)

##########
## Test ##
##########
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)

  ament_lint_auto_find_test_dependencies()



# if(CATKIN_ENABLE_TESTING)
#   find_package(catkin REQUIRED
#     COMPONENTS
#       ${CATKIN_PACKAGE_DEPENDENCIES}
#       grid_map_filters
#       roslaunch
#       rostest
#   )

#   # Cummulative distribution
#   catkin_add_gtest(test_${PROJECT_NAME}_cumulative_distribution
#     test/ElevationMapTest.cpp
#     test/test_elevation_mapping.cpp
#     test/WeightedEmpiricalCumulativeDistributionFunctionTest.cpp
#   )

#   target_link_libraries(test_${PROJECT_NAME}_cumulative_distribution
#     ${PROJECT_NAME}_library
#   )

#   target_include_directories(test_${PROJECT_NAME}_cumulative_distribution
#     PRIVATE
#       include
#   )

#   target_include_directories(test_${PROJECT_NAME}_cumulative_distribution
#     SYSTEM PUBLIC
#       ${catkin_INCLUDE_DIRS}
#   )

  # Input sources
  ament_add_gtest(test_${PROJECT_NAME}_input_sources
    test/input_sources/input_sources.test
    test/input_sources/test_input_sources.cpp
    test/input_sources/InputSourcesTest.cpp
  )

  target_link_libraries(test_${PROJECT_NAME}_input_sources
    ${PROJECT_NAME}_library
  )

  target_include_directories(test_${PROJECT_NAME}_input_sources
    PRIVATE
      include
  )

  target_include_directories(test_${PROJECT_NAME}_input_sources SYSTEM PUBLIC
    ${Boost_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS}
  )

#   # Postprocessor
#   add_rostest_gtest(test_${PROJECT_NAME}_postprocessor
#     test/postprocessing/postprocessor.test
#     test/postprocessing/main.cpp
#     test/postprocessing/PostprocessorTest.cpp
#   )

#   target_include_directories(test_${PROJECT_NAME}_postprocessor PRIVATE
#     include
#   )

#   target_include_directories(test_${PROJECT_NAME}_postprocessor SYSTEM PUBLIC
#     ${catkin_INCLUDE_DIRS}
#   )

#   target_link_libraries(test_${PROJECT_NAME}_postprocessor
#     ${GTEST_LIBRARIES}
#     ${PROJECT_NAME}_library
#     ${catkin_LIBRARIES}
#   )

#   find_package(cmake_code_coverage QUIET)
#   if(cmake_code_coverage_FOUND)
#       add_gtest_coverage(TEST_BUILD_TARGETS
#         test_${PROJECT_NAME}_cumulative_distribution
#       )
#       add_rostest_coverage(TEST_BUILD_TARGETS
#         test_${PROJECT_NAME}_input_sources
#       )
#       add_rostest_coverage(TEST_BUILD_TARGETS
#         test_${PROJECT_NAME}_postprocessor
#       )
#   endif()
endif()

###########
## Clang ##
###########

find_package(cmake_clang_tools QUIET)
if(cmake_clang_tools_FOUND)
  add_default_clang_tooling()
endif(cmake_clang_tools_FOUND)

ament_export_include_directories(include)
ament_export_libraries(
  ${PROJECT_NAME}_pcl_types
  ${PROJECT_NAME}_library
)
ament_export_dependencies(${DEPENDENCIES})


ament_package()
