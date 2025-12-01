cmake_minimum_required(VERSION 3.5)
project(kindr_ros)

# Use C++14 standard
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(tf2 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(kindr REQUIRED)

find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

set(dependencies
  eigen3_cmake_module
  Eigen3
  tf2
  geometry_msgs
  kindr
)

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION include/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(${PROJECT_NAME}-test
    test/test_main.cpp
    test/RosGeometryMsgPhysicalQuantitiesTest.cpp
    test/RosGeometryMsgRotationTest.cpp
    test/RosGeometryMsgPoseTest.cpp
    test/RosTfPoseTest.cpp
    test/TfConventionTest.cpp
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test
  )

  if(TARGET ${PROJECT_NAME}-test)
    target_include_directories(${PROJECT_NAME}-test PUBLIC
      include
      ${EIGEN3_INCLUDE_DIRS})
    ament_target_dependencies(${PROJECT_NAME}-test
      ${dependencies}
    )
    # TODO(SivertHavso): add back code coverage:
    #   find_package(cmake_code_coverage QUIET)
    #   if(cmake_code_coverage_FOUND)
    #     add_gtest_coverage(TEST_BUILD_TARGETS ${PROJECT_NAME}-test)
    #   endif()
  endif()
endif()

ament_export_include_directories(include)

ament_export_dependencies(
  ${dependencies}
)

ament_package()
