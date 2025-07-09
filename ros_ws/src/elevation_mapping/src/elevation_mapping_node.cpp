/*
 * elevation_map_node.cpp
 *
 *  Created on: Oct 3, 2013
 *      Author: Péter Fankhauser
 *   Institute: ETH Zurich, ANYbotics
 */ 

#include <rclcpp/rclcpp.hpp>
#include "elevation_mapping/ElevationMapping.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
     
  auto nodeHandle = rclcpp::Node::make_shared("elevation_mapping");

  nodeHandle->declare_parameter("num_callback_threads", 5);
  nodeHandle->declare_parameter("postprocessor_num_threads", 1);

  elevation_mapping::ElevationMapping elevationMap(nodeHandle);  

  // Spin
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), nodeHandle->get_parameter("num_callback_threads").as_int());
  executor.add_node(nodeHandle);
  executor.spin();
  //rclcpp::spin(nodeHandle);
  rclcpp::shutdown();  
  return 0;
}
