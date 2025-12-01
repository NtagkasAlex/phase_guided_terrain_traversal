#include "grid_map_filter_node/FiltersDemo.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<grid_map_filter_node::FiltersDemo>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

