#include "heightmap_node/heightmap_node.hpp"

#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>

HeightmapNode::HeightmapNode()
: rclcpp::Node("heightmap_node")
{
  using std::placeholders::_1;

  transformBuffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  transformListener_ = std::make_shared<tf2_ros::TransformListener>(*transformBuffer_);
  read_parameters();
  elevation_map_sub_ = this->create_subscription<grid_map_msgs::msg::GridMap>(
    input_name, 10, std::bind(&HeightmapNode::gridMapCallback, this, _1));

  height_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    output_name, 10);
  initialize();
}

void HeightmapNode::gridMapCallback(const grid_map_msgs::msg::GridMap::SharedPtr msg)
{
  grid_map::GridMap map;
  if (!grid_map::GridMapRosConverter::fromMessage(*msg, map)) {
    RCLCPP_WARN(this->get_logger(), "Failed to convert GridMap message.");
    return;
  }


  tf2::Stamped<tf2::Transform> transform;
  geometry_msgs::msg::TransformStamped transform_msg;

  transform_msg = transformBuffer_->lookupTransform("start", "base_link" , rclcpp::Time(0), rclcpp::Duration::from_seconds(5.0));
  tf2::fromMsg(transform_msg, transform);
  double roll, pitch, yaw;

  tf2::Matrix3x3(transform.getRotation()).getRPY(roll, pitch, yaw);
  

  Eigen::Vector3d position(transform.getOrigin().x(),
                         transform.getOrigin().y(),
                         transform.getOrigin().z());
  getHeightMap(map,position,-yaw);

  height_map_pub_->publish(cloud_msg);
}

void HeightmapNode::getHeightMap(grid_map::GridMap& map, const Eigen::Vector3d& position, double yaw){
    Eigen::Matrix2f R_W2H;
    R_W2H << std::cos(yaw),  std::sin(yaw),
            -std::sin(yaw),  std::cos(yaw);

    float c_h = (num_heightscans - 1) / 2.0f;
    float c_w = (num_widthscans - 1) / 2.0f;

    for (int i = 0; i < num_heightscans; ++i) {
        for (int j = 0; j < num_widthscans; ++j) {
            float dx = (c_h - i) * dist_x;
            float dy = (c_w - j) * dist_y;
            Eigen::Vector2f offset(dx, dy);
            offset = R_W2H * offset;

            Eigen::Vector2d query_pos(position.x() + offset.x(), position.y() + offset.y());

            grid_map::Position gridPos(query_pos.x(), query_pos.y());
            grid_map::Index index;

            float elevation = 0.0;//std::numeric_limits<float>::quiet_NaN();  // Default if index is invalid

            if (map.getIndex(gridPos, index)) {
                elevation = map.at(layer, index);
            }
            if (std::isnan(elevation)) {
                elevation = 0.0f;
            }
            size_t point_offset = i * cloud_msg.row_step + j * cloud_msg.point_step;

            float x = static_cast<float>(query_pos.x());
            float y = static_cast<float>(query_pos.y());
            float z = elevation;

            std::memcpy(&cloud_msg.data[point_offset + 0], &x, sizeof(float));
            std::memcpy(&cloud_msg.data[point_offset + sizeof(float)], &y, sizeof(float));
            std::memcpy(&cloud_msg.data[point_offset + 2 * sizeof(float)], &z, sizeof(float));
        }
    }
}
void HeightmapNode::read_parameters(){
  this->declare_parameter("num_heightscans",9);
  this->declare_parameter("num_widthscans",7);
  this->declare_parameter("dist_x",0.08);
  this->declare_parameter("dist_y",0.08);
  this->declare_parameter("output","elevation_heightmap");
  this->declare_parameter("input","elevation_map");
  this->declare_parameter("layer","elevation");

  this->get_parameter("num_heightscans",num_heightscans);
  this->get_parameter("num_widthscans",num_widthscans);
  this->get_parameter("dist_x",dist_x);
  this->get_parameter("dist_y",dist_y);
  this->get_parameter("input",input_name);
  this->get_parameter("output",output_name);
  this->get_parameter("layer",layer);

}
void HeightmapNode::initialize(){
    float dx = 0.1f;  // grid spacing in x-direction
    float dy = 0.1f;  // grid spacing in y-direction

    cloud_msg.header.stamp = rclcpp::Clock().now();
    cloud_msg.header.frame_id = "start";

    sensor_msgs::msg::PointField x_field;
    x_field.name = "x";
    x_field.offset = 0;
    x_field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    x_field.count = 1;

    sensor_msgs::msg::PointField y_field = x_field;
    y_field.name = "y";
    y_field.offset = sizeof(float);

    sensor_msgs::msg::PointField z_field = x_field;
    z_field.name = "z";
    z_field.offset = 2 * sizeof(float);

    cloud_msg.fields = {x_field, y_field, z_field};

    cloud_msg.height = num_heightscans;
    cloud_msg.width = num_widthscans;
    cloud_msg.is_bigendian = false;
    cloud_msg.point_step = 3 * sizeof(float);  // x, y, z
    cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;
    cloud_msg.is_dense = true;

    cloud_msg.data.resize(cloud_msg.row_step * cloud_msg.height);

    for (uint32_t row = 0; row < num_heightscans; ++row)
    {
        for (uint32_t col = 0; col < num_widthscans; ++col)
        {
            size_t point_offset = row * cloud_msg.row_step + col * cloud_msg.point_step;

            float x = col * dx;
            float y = row * dy;
            float z = static_cast<float>(row);  

            std::memcpy(&cloud_msg.data[point_offset + x_field.offset], &x, sizeof(float));
            std::memcpy(&cloud_msg.data[point_offset + y_field.offset], &y, sizeof(float));
            std::memcpy(&cloud_msg.data[point_offset + z_field.offset], &z, sizeof(float));
        }
    }
}
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HeightmapNode>());
  rclcpp::shutdown();
  return 0;
}

