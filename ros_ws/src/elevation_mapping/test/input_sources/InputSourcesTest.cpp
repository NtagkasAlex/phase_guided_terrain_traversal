/*
 *  InputSourceTest.cpp
 *
 *  Created on: Oct 02, 2020
 *  Author: Magnus Gärtner
 *  Institute: ETH Zurich, ANYbotics
 */

#include "elevation_mapping/ElevationMapping.hpp"

#include <rclcpp/rclcpp.hpp>

#include <gtest/gtest.h>

static void assertSuccessAndNumberOfSources(const std::string& inputConfiguration, bool successExpected,
                                            uint32_t numberOfExpectedInputSources) {
  elevation_mapping::InputSourceManager inputSourceManager(rclcpp::Node::make_shared("_"));
  bool success = inputSourceManager.configureFromRos(inputConfiguration);
  // ASSERT_EQ(success, successExpected) << "Configuration was:\n"
                                      // << ros::NodeHandle("~").param<XmlRpc::XmlRpcValue>(inputConfiguration, "not set").toXml() << "\n";
  ASSERT_EQ(inputSourceManager.getNumberOfSources(), numberOfExpectedInputSources);
}

TEST(InputSources, SingleInputValid) {  // NOLINT
  assertSuccessAndNumberOfSources("single_valid", true, 1);
}

TEST(InputSources, MultipleInputsValid) {  // NOLINT
  assertSuccessAndNumberOfSources("multiple_valid", true, 3);
}

TEST(InputSources, NoType) {  // NOLINT
  assertSuccessAndNumberOfSources("no_type", false, 0);
}

TEST(InputSources, NoTopic) {  // NOLINT
  assertSuccessAndNumberOfSources("no_topic", false, 0);
}

TEST(InputSources, NoQueueSize) {  // NOLINT
  assertSuccessAndNumberOfSources("no_queue_size", false, 0);
}

TEST(InputSources, NoPublishOnUpdate) {  // NOLINT
  assertSuccessAndNumberOfSources("no_publish_on_update", false, 0);
}

TEST(InputSources, SubscribingSameTwice) {  // NOLINT
  assertSuccessAndNumberOfSources("subscribing_same_topic_twice", false, 1);
}

TEST(InputSources, ConfigurationNotGiven) {  // NOLINT
  assertSuccessAndNumberOfSources("unset_namespace", false, 0);
}

TEST(InputSources, ConfigurationEmptySources) {  // NOLINT
  assertSuccessAndNumberOfSources("empty_sources_list", true, 0);
}

TEST(InputSources, ConfigurationWrongType) {  // NOLINT
  assertSuccessAndNumberOfSources("wrong_type_configuration", false, 0);
}

TEST(InputSources, ConfigurationNotAStruct) {  // NOLINT
  assertSuccessAndNumberOfSources("not_a_struct", false, 0);
}

TEST(InputSources, ConfigurationQueueSizeIsString) {  // NOLINT
  assertSuccessAndNumberOfSources("queue_size_is_string", false, 0);
}

TEST(InputSources, ConfigurationQueueSizeIsNegative) {  // NOLINT
  assertSuccessAndNumberOfSources("negative_queue_size", false, 0);
}

TEST(InputSources, UnknownType) {  // NOLINT
  auto nodeHandle =  rclcpp::Node::make_shared("_");
  elevation_mapping::InputSourceManager inputSourceManager(nodeHandle);
  inputSourceManager.configureFromRos("unknown_type");

  elevation_mapping::ElevationMapping map{nodeHandle};

  // Trying to register this misconfigured InputSourceManager to our map should fail.
  bool success =
      inputSourceManager.registerCallbacks(map, make_pair("pointcloud", &elevation_mapping::ElevationMapping::pointCloudCallback));
  ASSERT_FALSE(success);
}

TEST(ElevationMap, Constructor) {  // NOLINT
  auto nodeHandle =  rclcpp::Node::make_shared("_");
  elevation_mapping::ElevationMapping map(nodeHandle);
}

TEST(InputSources, ListeningToTopicsAfterRegistration) {  // NOLINT
  // subscribe to the default parameter "input_sources"
  auto nodeHandle =  rclcpp::Node::make_shared("_");
  class ElevationMappingWithInputSourcesAccessor : public elevation_mapping::ElevationMapping {
   public:
    ElevationMappingWithInputSourcesAccessor(std::shared_ptr<rclcpp::Node> nodeHandle) : elevation_mapping::ElevationMapping(nodeHandle) {}
    virtual ~ElevationMappingWithInputSourcesAccessor() = default;
    int getNumberOfSources() { return inputSources_.getNumberOfSources(); }
  } map{nodeHandle};

  // Wait a bit.
  rclcpp::spin_some(nodeHandle);
  rclcpp::sleep_for(std::chrono::seconds(1));
  rclcpp::spin_some(nodeHandle);

  // Publish to the topics we expect map to subscribe.
  auto nh = std::make_shared<rclcpp::Node>("_");
  auto firstLidarPublisher = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/lidar_1/depth/points", 1);
  auto secondLidarPublisher = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/lidar_2/depth/points", 1);

  // Check if we have exactly one subscriber per topic.
  ASSERT_EQ(firstLidarPublisher->get_subscription_count(), 1);
  ASSERT_EQ(secondLidarPublisher->get_subscription_count(), 1);
  // ASSERT_EQ(firstDepthImagePublisher.getNumSubscribers(), 1);
  ASSERT_EQ(map.getNumberOfSources(), 2);
}
