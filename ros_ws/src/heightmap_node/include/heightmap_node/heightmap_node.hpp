#ifndef HEIGHTMAP_NODE__HEIGHTMAP_NODE_HPP_
#define HEIGHTMAP_NODE__HEIGHTMAP_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <grid_map_core/GridMap.hpp>

#include <Eigen/Dense>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <iostream>
class HeightmapNode : public rclcpp::Node
{
public:
  HeightmapNode();
  void initialize();
  void read_parameters();
  void getHeightMap(grid_map::GridMap& map, const Eigen::Vector3d& position, double yaw);
private:
  void gridMapCallback(const grid_map_msgs::msg::GridMap::SharedPtr msg);

  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr elevation_map_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr height_map_pub_;
  sensor_msgs::msg::PointCloud2  cloud_msg;

  std::shared_ptr<tf2_ros::Buffer> transformBuffer_;
  std::shared_ptr<tf2_ros::TransformListener> transformListener_;

  int num_heightscans;
  int num_widthscans;
  double dist_x;
  double dist_y;
  std::string input_name;
  std::string output_name;
  std::string layer;

};

#endif  // HEIGHTMAP_NODE__HEIGHTMAP_NODE_HPP_

