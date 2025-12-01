#include "rclcpp/rclcpp.hpp"
#include <grid_map_msgs/srv/get_grid_map.hpp>

#include <chrono>
#include <cstdlib>
#include <memory>

using namespace std::chrono_literals;

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("get_grid_map_client");
  rclcpp::Client<grid_map_msgs::srv::GetGridMap>::SharedPtr client =
    node->create_client<grid_map_msgs::srv::GetGridMap>("/get_raw_submap");

  auto request = std::make_shared<grid_map_msgs::srv::GetGridMap::Request>();
  request->frame_id = "map";
  request->position_x = 0.0;
  request->position_y = 0.0;
  request->length_x = 10.0;
  request->length_y = 10.0;
  request->layers = {"elevation"};

  while (!client->wait_for_service(1s)) {
    if (!rclcpp::ok()) {
      RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Interrupted while waiting for the service. Exiting.");
      return 0;
    }
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "service not available, waiting again...");
  }

  auto result = client->async_send_request(request);
  // Wait for the result.
  if (rclcpp::spin_until_future_complete(node, result) ==
    rclcpp::FutureReturnCode::SUCCESS)
  {
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Grid map received.");
    auto response = result.get();
    // print the info of the grid map
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Grid map info: %s", response->map.header.frame_id.c_str());
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Grid map resolution: %f", response->map.info.resolution);
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Failed to get grid map.");
  }

  rclcpp::shutdown();
  return 0;
}