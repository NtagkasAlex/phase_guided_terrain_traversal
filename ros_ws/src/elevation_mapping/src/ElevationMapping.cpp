/*
 * ElevationMapping.cpp
 *
 *  Created on: Nov 12, 2013
 *      Author: Péter Fankhauser
 *   Institute: ETH Zurich, ANYbotics
 */

#define BOOST_BIND_NO_PLACEHOLDERS

#include <cmath>
#include <string>

#include <grid_map_msgs/msg/grid_map.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <boost/bind.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <grid_map_ros/grid_map_ros.hpp>
#include <kindr/Core>
#include <kindr_ros/kindr_ros.hpp>
#include <geometry_msgs/msg/transform_stamped.h>
#include <geometry_msgs/msg/point_stamped.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include "elevation_mapping/ElevationMap.hpp"
#include "elevation_mapping/ElevationMapping.hpp"
#include "elevation_mapping/PointXYZRGBConfidenceRatio.hpp"
#include "elevation_mapping/sensor_processors/LaserSensorProcessor.hpp"
#include "elevation_mapping/sensor_processors/PerfectSensorProcessor.hpp"
#include "elevation_mapping/sensor_processors/StereoSensorProcessor.hpp"
#include "elevation_mapping/sensor_processors/StructuredLightSensorProcessor.hpp"

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include <Eigen/Dense>

namespace elevation_mapping {

ElevationMapping::ElevationMapping(std::shared_ptr<rclcpp::Node>& nodeHandle) :
      nodeHandle_(nodeHandle),
      inputSources_(nodeHandle_),
      robotPoseCacheSize_(200),
      // transformListener_(transformBuffer_),
      map_(nodeHandle),
      robotMotionMapUpdater_(nodeHandle),
      ignoreRobotMotionUpdates_(false),
      updatesEnabled_(true),                    
      maxNoUpdateDuration_(rclcpp::Duration::from_seconds(0.0)),
      timeTolerance_(rclcpp::Duration::from_seconds(0.0)),
      fusedMapPublishTimerDuration_(rclcpp::Duration::from_seconds(0.0)),
      isContinuouslyFusing_(false),
      visibilityCleanupTimerDuration_(rclcpp::Duration::from_seconds(0.0)),
      receivedFirstMatchingPointcloudAndPose_(false),  
      initializeElevationMap_(true),
      initializationMethod_(0),
      lengthInXInitSubmap_(1.2),
      lengthInYInitSubmap_(1.8),
      marginInitSubmap_(0.3),  
      initSubmapHeightOffset_(0.0)
       {
#ifndef NDEBUG
  // Print a warning if built in debug.
  RCLCPP_WARN(nodeHandle_->get_logger(), "CMake Build Type is 'Debug'. Change to 'Release' for better performance.");
#endif

  RCLCPP_INFO(nodeHandle_->get_logger(), "Elevation mapping node started.");

  readParameters();
  setupSubscribers();
  setupServices();
  setupTimers();

  transformBuffer_ = std::make_shared<tf2_ros::Buffer>(nodeHandle_->get_clock());
  transformListener_ = std::make_shared<tf2_ros::TransformListener>(*transformBuffer_);
  
  initialize();

  RCLCPP_INFO(nodeHandle_->get_logger(), "Successfully launched node.");
}

void ElevationMapping::setupSubscribers() {  // Handle deprecated point_cloud_topic and input_sources configuration.
  auto res = nodeHandle_->get_topic_names_and_types();
  for (auto a:res){
    RCLCPP_INFO(nodeHandle_->get_logger(), "topic: %s", a.first.c_str());
  }

  const bool configuredInputSources = inputSources_.configureFromRos("input_sources");
  const bool hasDeprecatedPointcloudTopic = nodeHandle_->get_parameter("point_cloud_topic", pointCloudTopic_);
  if (hasDeprecatedPointcloudTopic) {
    RCLCPP_WARN(nodeHandle_->get_logger(), "Parameter 'point_cloud_topic' is deprecated, please use 'input_sources' instead.");
  }
  /*if (!configuredInputSources && hasDeprecatedPointcloudTopic) {
    pointCloudSubscriber_ = nodeHandle_->create_subscription<sensor_msgs::msg::PointCloud2>(        
        pointCloudTopic_, 1, [&](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {pointCloudCallback(msg, true, sensorProcessor_);});
  }*/       
  if (configuredInputSources) {
    inputSources_.registerCallbacks(*this, std::make_pair("pointcloud", &ElevationMapping::pointCloudCallback));
    // inputSources_.registerCallbacks(*this, std::make_pair("pointcloud", pointCloudCallback));
  } else {
    RCLCPP_ERROR(nodeHandle_->get_logger(), "Input sources not configured!");
  }
  
  if (!robotPoseTopic_.empty()) {
    robotPoseSubscriber_.subscribe(nodeHandle_, robotPoseTopic_,qos_profile);
    robotPoseCache_.connectInput(robotPoseSubscriber_);
    robotPoseCache_.setCacheSize(robotPoseCacheSize_);
  } else {
    ignoreRobotMotionUpdates_ = true;
  }
}

void ElevationMapping::setupServices() {
  // Multi-threading for fusion.  
  fusionTriggerService_ = nodeHandle_->create_service<std_srvs::srv::Empty>("trigger_fusion", std::bind(&ElevationMapping::fuseEntireMapServiceCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), rclcpp::ServicesQoS().get_rmw_qos_profile(), fusionServiceGroup_);
  
  fusedSubmapService_ = nodeHandle_->create_service<grid_map_msgs::srv::GetGridMap>("get_submap", std::bind(&ElevationMapping::getFusedSubmapServiceCallback, this, std::placeholders::_1, std::placeholders::_2), rclcpp::ServicesQoS().get_rmw_qos_profile(), fusionServiceGroup_);

  rawSubmapService_ = nodeHandle_->create_service<grid_map_msgs::srv::GetGridMap>("get_raw_submap", std::bind(&ElevationMapping::getRawSubmapServiceCallback, this, std::placeholders::_1, std::placeholders::_2), rclcpp::ServicesQoS().get_rmw_qos_profile(), fusionServiceGroup_);

  clearMapService_ = nodeHandle_->create_service<std_srvs::srv::Empty>("clear_map", std::bind(&ElevationMapping::clearMapServiceCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  enableUpdatesService_ = nodeHandle_->create_service<std_srvs::srv::Empty>("enable_updates", std::bind(&ElevationMapping::enableUpdatesServiceCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  disableUpdatesService_ = nodeHandle_->create_service<std_srvs::srv::Empty>("disable_updates", std::bind(&ElevationMapping::disableUpdatesServiceCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  maskedReplaceService_ = nodeHandle_->create_service<grid_map_msgs::srv::SetGridMap>("masked_replace", std::bind(&ElevationMapping::maskedReplaceServiceCallback, this, std::placeholders::_1, std::placeholders::_2));
  saveMapService_ = nodeHandle_->create_service<grid_map_msgs::srv::ProcessFile>("save_map", std::bind(&ElevationMapping::saveMapServiceCallback, this, std::placeholders::_1, std::placeholders::_2));
  loadMapService_ = nodeHandle_->create_service<grid_map_msgs::srv::ProcessFile>("load_map", std::bind(&ElevationMapping::loadMapServiceCallback, this, std::placeholders::_1, std::placeholders::_2));
}

void ElevationMapping::setupTimers() {
  // TODO: mapUpdateTimer_ is originally single shot and autostart is set to false
  // mapUpdateTimer_ = rclcpp::create_timer(nodeHandle_, nodeHandle_->get_clock(), maxNoUpdateDuration_, &ElevationMapping::mapUpdateTimerCallback);

  if (fusedMapPublishTimerDuration_.seconds() != 0.0) {
    fusedMapPublishTimer_ = rclcpp::create_timer(
      nodeHandle_, nodeHandle_->get_clock(), fusedMapPublishTimerDuration_,
      std::bind(&ElevationMapping::publishFusedMapCallback, this));
  }

  // Multi-threading for visibility cleanup. Visibility clean-up does not help when continuous clean-up is enabled.
  if (map_.enableVisibilityCleanup_ && (visibilityCleanupTimerDuration_.seconds() != 0.0) && !map_.enableContinuousCleanup_) {
    visibilityCleanupTimer_ = rclcpp::create_timer(
      nodeHandle_, nodeHandle_->get_clock(), visibilityCleanupTimerDuration_,
      std::bind(&ElevationMapping::visibilityCleanupCallback, this));
    // visibilityCleanupTimer_->cancel(); //TODO:foxy does not have timer autostart flag implemented, fix in future version
    //TODO: timer reset issue: https://github.com/ros2/rclcpp/issues/1012
    //TODO: run cleanup in a separate thread   
  }
}

ElevationMapping::~ElevationMapping() {
  // Shutdown all services.

  {  // Fusion Service Queue
    rawSubmapService_.reset();
    fusionTriggerService_.reset();
    fusedSubmapService_.reset();
    fusedMapPublishTimer_->cancel();

    // fusionServiceQueue_.disable();
    // fusionServiceQueue_.clear();
  }

  {  // Visibility cleanup queue
    visibilityCleanupTimer_->cancel();

    // visibilityCleanupQueue_.disable();
    // visibilityCleanupQueue_.clear();
  }

  rclcpp::shutdown();

  // Join threads.
  /*if (fusionServiceThread_.joinable()) {
    fusionServiceThread_.join();
  }*/
  /*if (visibilityCleanupThread_.joinable()) {
    visibilityCleanupThread_.join();
  }*/
}

bool ElevationMapping::readParameters() {
  // ElevationMapping parameters.
  nodeHandle_->declare_parameter("point_cloud_topic","");
  //FIXME: Fix for case when robot pose is not defined
  nodeHandle_->declare_parameter("robot_pose_with_covariance_topic", std::string("/pose"));
  nodeHandle_->declare_parameter("track_point_frame_id", std::string("/robot"));
  nodeHandle_->declare_parameter("track_point_x", 0.0);
  nodeHandle_->declare_parameter("track_point_y", 0.0);
  nodeHandle_->declare_parameter("track_point_z", 0.0);
  nodeHandle_->declare_parameter("robot_pose_cache_size", 200);

  // nodeHandle_->get_parameter("point_cloud_topic", pointCloudTopic_);
  nodeHandle_->get_parameter("robot_pose_with_covariance_topic", robotPoseTopic_);  
  nodeHandle_->get_parameter("track_point_frame_id", trackPointFrameId_);
  nodeHandle_->get_parameter("track_point_x", trackPoint_.x());
  nodeHandle_->get_parameter("track_point_y", trackPoint_.y());
  nodeHandle_->get_parameter("track_point_z", trackPoint_.z());
  nodeHandle_->get_parameter("robot_pose_cache_size", robotPoseCacheSize_);

  assert(robotPoseCacheSize_ >= 0);

  double minUpdateRate;
  nodeHandle_->declare_parameter("min_update_rate", 2.0);
  nodeHandle_->get_parameter("min_update_rate", minUpdateRate);
  if (minUpdateRate == 0.0) {
    maxNoUpdateDuration_ = rclcpp::Duration::from_seconds(0);
    RCLCPP_WARN(nodeHandle_->get_logger(), "Rate for publishing the map is zero.");
  } else {
    maxNoUpdateDuration_ = rclcpp::Duration::from_seconds(1.0 / minUpdateRate);
  }
  assert(maxNoUpdateDuration_.seconds() != 0.0);

  double timeTolerance;
  nodeHandle_->declare_parameter("time_tolerance", 0.0);
  nodeHandle_->get_parameter("time_tolerance", timeTolerance);
  timeTolerance_ = rclcpp::Duration::from_seconds(timeTolerance);

  double fusedMapPublishingRate;
  nodeHandle_->declare_parameter("fused_map_publishing_rate", 1.0);
  nodeHandle_->get_parameter("fused_map_publishing_rate", fusedMapPublishingRate);
  if (fusedMapPublishingRate == 0.0) {
    fusedMapPublishTimerDuration_ = rclcpp::Duration::from_seconds(0.0);
    RCLCPP_WARN(nodeHandle_->get_logger(), 
        "Rate for publishing the fused map is zero. The fused elevation map will not be published unless the service `triggerFusion` is "
        "called.");
  } else if (std::isinf(fusedMapPublishingRate)) {
    isContinuouslyFusing_ = true;
    fusedMapPublishTimerDuration_ = rclcpp::Duration::from_seconds(0.0);
  } else {
    fusedMapPublishTimerDuration_ = rclcpp::Duration::from_seconds(1.0 / fusedMapPublishingRate);
  }

  double visibilityCleanupRate;
  nodeHandle_->declare_parameter("visibility_cleanup_rate", 1.0);
  nodeHandle_->get_parameter("visibility_cleanup_rate", visibilityCleanupRate);
  if (visibilityCleanupRate == 0.0) {
    visibilityCleanupTimerDuration_ = rclcpp::Duration::from_seconds(0.0);
    RCLCPP_WARN(nodeHandle_->get_logger(), "Rate for visibility cleanup is zero and therefore disabled.");
  } else {
    visibilityCleanupTimerDuration_ = rclcpp::Duration::from_seconds(1.0 / visibilityCleanupRate);
    map_.visibilityCleanupDuration_ = 1.0 / visibilityCleanupRate;
  }

  // ElevationMap parameters. TODO Move this to the elevation map class.
  nodeHandle_->declare_parameter("map_frame_id", std::string("/map"));
  nodeHandle_->get_parameter("map_frame_id", mapFrameId_);
  map_.setFrameId(mapFrameId_);

  grid_map::Length length;
  grid_map::Position position;
  double resolution;

  nodeHandle_->declare_parameter("length_in_x", 1.5);
  nodeHandle_->declare_parameter("length_in_y", 1.5);
  nodeHandle_->declare_parameter("position_x",  0.0);
  nodeHandle_->declare_parameter("position_y",  0.0);
  nodeHandle_->declare_parameter("resolution",  0.01);

  nodeHandle_->get_parameter("length_in_x", length(0));
  nodeHandle_->get_parameter("length_in_y", length(1));
  nodeHandle_->get_parameter("position_x", position.x());
  nodeHandle_->get_parameter("position_y", position.y());
  nodeHandle_->get_parameter("resolution", resolution);
  map_.setGeometry(length, resolution, position);

  nodeHandle_->declare_parameter("min_variance", pow(0.003, 2));
  nodeHandle_->declare_parameter("max_variance", pow(0.03, 2));
  nodeHandle_->declare_parameter("mahalanobis_distance_threshold", 2.5);
  nodeHandle_->declare_parameter("multi_height_noise", pow(0.003, 2));
  nodeHandle_->declare_parameter("min_horizontal_variance", pow(resolution / 2.0, 2));  // two-sigma
  nodeHandle_->declare_parameter("max_horizontal_variance", 0.5);
  nodeHandle_->declare_parameter("underlying_map_topic", std::string());
  nodeHandle_->declare_parameter("enable_visibility_cleanup", true);
  nodeHandle_->declare_parameter("enable_continuous_cleanup", false);
  nodeHandle_->declare_parameter("scanning_duration", 1.0);
  nodeHandle_->declare_parameter("masked_replace_service_mask_layer_name", std::string("mask"));

  nodeHandle_->get_parameter("min_variance", map_.minVariance_);
  nodeHandle_->get_parameter("max_variance", map_.maxVariance_);
  nodeHandle_->get_parameter("mahalanobis_distance_threshold", map_.mahalanobisDistanceThreshold_);
  nodeHandle_->get_parameter("multi_height_noise", map_.multiHeightNoise_);
  nodeHandle_->get_parameter("min_horizontal_variance", map_.minHorizontalVariance_);  // two-sigma
  nodeHandle_->get_parameter("max_horizontal_variance", map_.maxHorizontalVariance_);
  nodeHandle_->get_parameter("underlying_map_topic", map_.underlyingMapTopic_);
  nodeHandle_->get_parameter("enable_visibility_cleanup", map_.enableVisibilityCleanup_);
  nodeHandle_->get_parameter("enable_continuous_cleanup", map_.enableContinuousCleanup_);
  nodeHandle_->get_parameter("scanning_duration", map_.scanningDuration_);
  nodeHandle_->get_parameter("masked_replace_service_mask_layer_name", maskedReplaceServiceMaskLayerName_);

  // Settings for initializing elevation map
  nodeHandle_->declare_parameter("initialize_elevation_map", false);
  nodeHandle_->declare_parameter("initialization_method", 0);
  nodeHandle_->declare_parameter("length_in_x_init_submap", 1.2);
  nodeHandle_->declare_parameter("length_in_y_init_submap", 1.8);
  nodeHandle_->declare_parameter("margin_init_submap", 0.3);
  nodeHandle_->declare_parameter("init_submap_height_offset", 0.0);
  nodeHandle_->declare_parameter("target_frame_init_submap", std::string("/footprint"));

  nodeHandle_->get_parameter("initialize_elevation_map", initializeElevationMap_);
  nodeHandle_->get_parameter("initialization_method", initializationMethod_);
  nodeHandle_->get_parameter("length_in_x_init_submap", lengthInXInitSubmap_);
  nodeHandle_->get_parameter("length_in_y_init_submap", lengthInYInitSubmap_);
  nodeHandle_->get_parameter("margin_init_submap", marginInitSubmap_);
  nodeHandle_->get_parameter("init_submap_height_offset", initSubmapHeightOffset_);
  nodeHandle_->get_parameter("target_frame_init_submap", targetFrameInitSubmap_);

  nodeHandle_->declare_parameter("robot_base_frame_id", std::string("/robot"));
  
  nodeHandle_->declare_parameter("num_heightscans",9);
  nodeHandle_->declare_parameter("num_widthscans",7);
  nodeHandle_->declare_parameter("dist_x",0.08);
  nodeHandle_->declare_parameter("dist_y",0.08);

  nodeHandle_->get_parameter("num_heightscans",map_.num_heightscans);
  nodeHandle_->get_parameter("num_widthscans",map_.num_widthscans);
  nodeHandle_->get_parameter("dist_x",map_.dist_x);
  nodeHandle_->get_parameter("dist_y",map_.dist_y);

  nodeHandle_->declare_parameter("increase_height_alpha",1.0);
  nodeHandle_->get_parameter("increase_height_alpha",map_.increaseHeightAlpha_);


  // SensorProcessor parameters. Deprecated, use the sensorProcessor from within input sources instead!
  /*\\std::string sensorType;
  nodeHandle_->declare_parameter("sensor_processor/type", std::string("structured_light"));
  nodeHandle_->get_parameter("sensor_processor/type", sensorType);
  
  SensorProcessorBase::GeneralParameters generalSensorProcessorConfig{nodeHandle_->get_parameter("robot_base_frame_id").as_string(), mapFrameId_};
  if (sensorType == "structured_light") {
    sensorProcessor_.reset(new StructuredLightSensorProcessor(nodeHandle_, generalSensorProcessorConfig)); // ERROR
  } else if (sensorType == "stereo") {
    sensorProcessor_.reset(new StereoSensorProcessor(nodeHandle_, generalSensorProcessorConfig));
  } else if (sensorType == "laser") {
    sensorProcessor_.reset(new LaserSensorProcessor(nodeHandle_, generalSensorProcessorConfig));
  } else if (sensorType == "perfect") {
    sensorProcessor_.reset(new PerfectSensorProcessor(nodeHandle_, generalSensorProcessorConfig));
  } else {
    RCLCPP_ERROR(nodeHandle_->get_logger(), "The sensor type %s is not available.", sensorType.c_str());
  }
  if (!sensorProcessor_->readParameters()) {
    return false;
  }*/
  if (!robotMotionMapUpdater_.readParameters()) {
    return false;
  }

  return true;
}

bool ElevationMapping::initialize() {
  RCLCPP_INFO(nodeHandle_->get_logger(), "Elevation mapping node initializing ... ");

  fusionServiceGroup_ = nodeHandle_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  //fusionServiceThread_ = std::thread(boost::bind(&ElevationMapping::runFusionServiceThread, this));
  rclcpp::sleep_for(std::chrono::seconds(1));  // Need this to get the TF caches fill up.
  // resetMapUpdateTimer();
  fusedMapPublishTimer_->reset();
  // visibilityCleanupThread_ = boost::thread(boost::bind(&ElevationMapping::visibilityCleanupThread, this));
  // visibilityCleanupTimer_.reset();  //TODO:foxy does not have timer autostart flag implemented, fix in future version
  initializeElevationMap();
  // === 1. Set header ===

  float dx = 0.1f;  // grid spacing in x-direction
float dy = 0.1f;  // grid spacing in y-direction

// Header
map_.heightmap_msg.header.stamp = rclcpp::Clock().now();
map_.heightmap_msg.header.frame_id = mapFrameId_;

// Define PointFields for x, y, z
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

map_.heightmap_msg.fields = {x_field, y_field, z_field};

// Image-like layout
map_.heightmap_msg.height = map_.num_heightscans;
map_.heightmap_msg.width = map_.num_widthscans;
map_.heightmap_msg.is_bigendian = false;
map_.heightmap_msg.point_step = 3 * sizeof(float);  // x, y, z
map_.heightmap_msg.row_step = map_.heightmap_msg.point_step * map_.heightmap_msg.width;
map_.heightmap_msg.is_dense = true;

// Allocate data buffer
map_.heightmap_msg.data.resize(map_.heightmap_msg.row_step * map_.heightmap_msg.height);

// Fill point cloud data
for (uint32_t row = 0; row < map_.num_heightscans; ++row)
{
    for (uint32_t col = 0; col < map_.num_widthscans; ++col)
    {
        size_t point_offset = row * map_.heightmap_msg.row_step + col * map_.heightmap_msg.point_step;

        float x = col * dx;
        float y = row * dy;
        float z = static_cast<float>(row);  // or any other z value

        std::memcpy(&map_.heightmap_msg.data[point_offset + x_field.offset], &x, sizeof(float));
        std::memcpy(&map_.heightmap_msg.data[point_offset + y_field.offset], &y, sizeof(float));
        std::memcpy(&map_.heightmap_msg.data[point_offset + z_field.offset], &z, sizeof(float));
    }
}

  return true;
}

/*void ElevationMapping::runFusionServiceThread() {
  rclcpp::Rate loopRate(20);

  while (rclcpp::ok()) {
    fusionServiceQueue_.callAvailable();

    // Sleep until the next execution.
    loopRate.sleep();
  }
}*/

/*void ElevationMapping::visibilityCleanupThread() {
  rclcpp::Rate loopRate(20);

  while (rclcpp::ok()) {
    visibilityCleanupQueue_.callAvailable();

    // Sleep until the next execution.
    loopRate.sleep();
  }
}*/
 
void ElevationMapping::pointCloudCallback(sensor_msgs::msg::PointCloud2::ConstSharedPtr pointCloudMsg, bool publishPointCloud,
                                          const SensorProcessorBase::Ptr& sensorProcessor_) {
  // RCLCPP_INFO(nodeHandle_->get_logger(), "Processing data from: %s", pointCloudMsg->header.frame_id.c_str());
  if (!updatesEnabled_) {
    auto clock = nodeHandle_->get_clock();
    RCLCPP_WARN_THROTTLE(nodeHandle_->get_logger(), *(clock), 10, "Updating of elevation map is disabled. (Warning message is throttled, 10s.)");
    if (publishPointCloud) {
      map_.setTimestamp(nodeHandle_->get_clock()->now());
      map_.postprocessAndPublishRawElevationMap();
    }
    return;
  }

  // Check if point cloud has corresponding robot pose at the beginning
  if (!receivedFirstMatchingPointcloudAndPose_) {
    const double oldestPoseTime = robotPoseCache_.getOldestTime().seconds();
    const double currentPointCloudTime = rclcpp::Time(pointCloudMsg->header.stamp,RCL_ROS_TIME).seconds();

    if (currentPointCloudTime < oldestPoseTime) {
      auto clock = nodeHandle_->get_clock();
      RCLCPP_WARN_THROTTLE(nodeHandle_->get_logger(), *(clock), 5, "No corresponding point cloud and pose are found. Waiting for first match. (Warning message is throttled, 5s.)");
      return;
    } else {
      RCLCPP_INFO(nodeHandle_->get_logger(), "First corresponding point cloud and pose found, elevation mapping started. ");
      receivedFirstMatchingPointcloudAndPose_ = true;
    }
  }

  // stopMapUpdateTimer();

  // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud.
  // TODO(max): Double check with http://wiki.ros.org/hydro/Migration
  pcl::PCLPointCloud2 pcl_pc;
  pcl_conversions::toPCL(*pointCloudMsg, pcl_pc);

  PointCloudType::Ptr pointCloud(new PointCloudType);
  pcl::fromPCLPointCloud2(pcl_pc, *pointCloud);
  lastPointCloudUpdateTime_ = rclcpp::Time(1000 * pointCloud->header.stamp, RCL_ROS_TIME);

  // RCLCPP_INFO(nodeHandle_->get_logger(), "ElevationMap received a point cloud (%i points) for elevation mapping.", static_cast<int>(pointCloud->size()));
  RCLCPP_DEBUG(nodeHandle_->get_logger(), "ElevationMap received a point cloud (%i points) for elevation mapping.", static_cast<int>(pointCloud->size()));

  // Get robot pose covariance matrix at timestamp of point cloud.
  Eigen::Matrix<double, 6, 6> robotPoseCovariance;
  robotPoseCovariance.setZero();
  if (!ignoreRobotMotionUpdates_) {
    std::shared_ptr<const nav_msgs::msg::Odometry> poseMessage = robotPoseCache_.getElemBeforeTime(lastPointCloudUpdateTime_);
    if (!poseMessage) {
      // Tell the user that either for the timestamp no pose is available or that the buffer is possibly empty
      if (robotPoseCache_.getOldestTime().seconds() > lastPointCloudUpdateTime_.seconds()) {
        RCLCPP_ERROR(nodeHandle_->get_logger(), "The oldest pose available is at %f, requested pose at %f", robotPoseCache_.getOldestTime().seconds(),
                  lastPointCloudUpdateTime_.seconds());
      } else {
        RCLCPP_ERROR(nodeHandle_->get_logger(), "Could not get pose information from robot for time %f. Buffer empty?", lastPointCloudUpdateTime_.seconds());
      }
      return;
    }
    //std::cout<<poseMessage->pose.covariance.data()<<std::endl;
    robotPoseCovariance = Eigen::Map<const Eigen::MatrixXd>(poseMessage->pose.covariance.data(), 6, 6);
  //robotPoseCovariance.Zero();
  }

  // Process point cloud.
  // RCLCPP_INFO_STREAM(nodeHandle_->get_logger(),"Variance is:"<<robotPoseCovariance);
  PointCloudType::Ptr pointCloudProcessed(new PointCloudType);
  Eigen::VectorXf measurementVariances;
  if (!sensorProcessor_->process(pointCloud, robotPoseCovariance, pointCloudProcessed, measurementVariances,
                                 pointCloudMsg->header.frame_id)) {
    if (!sensorProcessor_->isTfAvailableInBuffer()) {
      rclcpp::Clock clock;
      RCLCPP_INFO_THROTTLE(nodeHandle_->get_logger(), clock, 10, "Waiting for tf transformation to be available. (Message is throttled, 10s.)");
      return;
    }
    RCLCPP_ERROR(nodeHandle_->get_logger(), "Point cloud could not be processed."); //TODO: what causes this issue
    // resetMapUpdateTimer();
    return;
  }
  
  // RCLCPP_INFO_STREAM(nodeHandle_->get_logger(),"Variance is:"<<measurementVariances[0]<<" , "<<measurementVariances[10]<<" , "<<measurementVariances[100]<<" , "<<measurementVariances[110]<<" , "<<measurementVariances[120]);

  boost::recursive_mutex::scoped_lock scopedLock(map_.getRawDataMutex());

  // Update map location.
  updateMapLocation();

  // Update map from motion prediction.
  if (!updatePrediction(lastPointCloudUpdateTime_)) {
    RCLCPP_ERROR(nodeHandle_->get_logger(), "Updating process noise failed.");
    // resetMapUpdateTimer();
    return;
  }

  // Clear the map if continuous clean-up was enabled.
  if (map_.enableContinuousCleanup_) {
    RCLCPP_DEBUG(nodeHandle_->get_logger(), "Clearing elevation map before adding new point cloud.");
    map_.clear();
  }

  // Add point cloud to elevation map.
  if (!map_.add(pointCloudProcessed, measurementVariances, lastPointCloudUpdateTime_,
                Eigen::Affine3d(sensorProcessor_->transformationSensorToMap_))) {
    RCLCPP_ERROR(nodeHandle_->get_logger(), "Adding point cloud to elevation map failed.");
    // resetMapUpdateTimer();
    return;
  }

  if (publishPointCloud) {
    RCLCPP_DEBUG(nodeHandle_->get_logger(), "Publishing pcl.");
    // Publish elevation map.
    map_.postprocessAndPublishRawElevationMap();
    if (isFusingEnabled()) {
      map_.fuseAll();
      map_.publishFusedElevationMap();
    }
  }

  // resetMapUpdateTimer();
}

void ElevationMapping::mapUpdateTimerCallback() {
  if (!updatesEnabled_) {
    rclcpp::Clock clock;
    RCLCPP_WARN_THROTTLE(nodeHandle_->get_logger(), clock, 10, "Updating of elevation map is disabled. (Warning message is throttled, 10s.)");
    map_.setTimestamp(nodeHandle_->get_clock()->now());
    map_.postprocessAndPublishRawElevationMap();
    return;
  }

  rclcpp::Time time = rclcpp::Clock(RCL_ROS_TIME).now();
  if ((lastPointCloudUpdateTime_ - time) <= maxNoUpdateDuration_) {  // there were updates from sensordata, no need to force an update.
    return;
  }
  rclcpp::Clock clock;
  RCLCPP_WARN_THROTTLE(nodeHandle_->get_logger(), clock, 5, "Elevation map is updated without data from the sensor. (Warning message is throttled, 5s.)");

  boost::recursive_mutex::scoped_lock scopedLock(map_.getRawDataMutex());

  // stopMapUpdateTimer();

  // Update map from motion prediction.
  if (!updatePrediction(time)) {
    RCLCPP_ERROR(nodeHandle_->get_logger(), "Updating process noise failed.");
    // resetMapUpdateTimer();
    return;
  }

  // Publish elevation map.
  map_.postprocessAndPublishRawElevationMap();
  if (isFusingEnabled()) {
    map_.fuseAll();
    map_.publishFusedElevationMap();
  }

  // resetMapUpdateTimer();
}

void ElevationMapping::publishFusedMapCallback() {
  if (!map_.hasFusedMapSubscribers()) {
    return;
  }
  RCLCPP_DEBUG(nodeHandle_->get_logger(), "Elevation map is fused and published from timer.");

  boost::recursive_mutex::scoped_lock scopedLock(map_.getFusedDataMutex());
  map_.fuseAll();

  tf2::Stamped<tf2::Transform> transform;
  geometry_msgs::msg::TransformStamped transform_msg;

  transform_msg = transformBuffer_->lookupTransform(mapFrameId_, trackPointFrameId_ , rclcpp::Time(0), rclcpp::Duration::from_seconds(5.0));
  tf2::fromMsg(transform_msg, transform);
  double roll, pitch, yaw;

  tf2::Matrix3x3(transform.getRotation()).getRPY(roll, pitch, yaw);
  
  // RCLCPP_INFO(nodeHandle_->get_logger(),"YAW: %.3f",yaw);

  Eigen::Vector3d position(transform.getOrigin().x(),
                         transform.getOrigin().y(),
                         transform.getOrigin().z());
  map_.getHeightMap(position,-yaw);

  map_.publishFusedElevationMap();
}

void ElevationMapping::visibilityCleanupCallback() {
  RCLCPP_DEBUG(nodeHandle_->get_logger(), "Elevation map is running visibility cleanup.");
  // Copy constructors for thread-safety.
  map_.visibilityCleanup(rclcpp::Time(lastPointCloudUpdateTime_,RCL_ROS_TIME));
} 

bool ElevationMapping::fuseEntireMapServiceCallback(const std::shared_ptr<rmw_request_id_t>, const std::shared_ptr<std_srvs::srv::Empty::Request>, std::shared_ptr<std_srvs::srv::Empty::Response>) {


  boost::recursive_mutex::scoped_lock scopedLock(map_.getFusedDataMutex());
  

  map_.fuseAll();
  // map_.getHeightMap();
  map_.publishFusedElevationMap();



  return true;
}

bool ElevationMapping::isFusingEnabled() {
  return isContinuouslyFusing_ && map_.hasFusedMapSubscribers();
}

bool ElevationMapping::updatePrediction(const rclcpp::Time& time) {
  if (ignoreRobotMotionUpdates_) {
    return true;
  }

  RCLCPP_DEBUG(nodeHandle_->get_logger(), "Updating map with latest prediction from time %f.", robotPoseCache_.getLatestTime().seconds());

  if (time + timeTolerance_ < map_.getTimeOfLastUpdate()) {
    RCLCPP_ERROR(nodeHandle_->get_logger(), "Requested update with time stamp %f, but time of last update was %f.", time.seconds(), map_.getTimeOfLastUpdate().seconds());
    return false;
  } else if (time < map_.getTimeOfLastUpdate()) {
    RCLCPP_DEBUG(nodeHandle_->get_logger(), "Requested update with time stamp %f, but time of last update was %f. Ignoring update.", time.seconds(),
              map_.getTimeOfLastUpdate().seconds());
    return true;
  }

  // Get robot pose at requested time.
  std::shared_ptr<const nav_msgs::msg::Odometry> poseMessage = robotPoseCache_.getElemBeforeTime(time);
  if (!poseMessage) {
    // Tell the user that either for the timestamp no pose is available or that the buffer is possibly empty
    if (robotPoseCache_.getOldestTime().seconds() > lastPointCloudUpdateTime_.seconds()) {
      RCLCPP_ERROR(nodeHandle_->get_logger(), "The oldest pose available is at %f, requested pose at %f", robotPoseCache_.getOldestTime().seconds(),
                lastPointCloudUpdateTime_.seconds());
    } else {
      RCLCPP_ERROR(nodeHandle_->get_logger(), "Could not get pose information from robot for time %f. Buffer empty?", lastPointCloudUpdateTime_.seconds());
    }
    return false;
  }

  kindr::HomTransformQuatD robotPose;
  kindr_ros::convertFromRosGeometryMsg(poseMessage->pose.pose, robotPose);
  // Covariance is stored in row-major in ROS: http://docs.ros.org/api/geometry_msgs/html/msg/PoseWithCovariance.html
  Eigen::Matrix<double, 6, 6> robotPoseCovariance =
      Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(poseMessage->pose.covariance.data(), 6, 6);

  // Compute map variance update from motion prediction.
  robotMotionMapUpdater_.update(map_, robotPose, robotPoseCovariance, time);

  return true;
}

bool ElevationMapping::updateMapLocation() {
  RCLCPP_DEBUG(nodeHandle_->get_logger(), "Elevation map is checked for relocalization.");

  geometry_msgs::msg::PointStamped trackPoint;
  trackPoint.header.frame_id = trackPointFrameId_;
  trackPoint.header.stamp = rclcpp::Time(0);
  kindr_ros::convertToRosGeometryMsg(trackPoint_, trackPoint.point);
  geometry_msgs::msg::PointStamped trackPointTransformed;

  try {
    trackPointTransformed = transformBuffer_->transform(trackPoint, map_.getFrameId());
  } catch (tf2::TransformException& ex) {
    RCLCPP_ERROR(nodeHandle_->get_logger(), "%s", ex.what());
    return false;
  }

  kindr::Position3D position3d;
  kindr_ros::convertFromRosGeometryMsg(trackPointTransformed.point, position3d);
  grid_map::Position position = position3d.vector().head(2);
  map_.move(position);
  return true;
}

bool ElevationMapping::getFusedSubmapServiceCallback(std::shared_ptr<grid_map_msgs::srv::GetGridMap::Request> request,
                                                     std::shared_ptr<grid_map_msgs::srv::GetGridMap::Response> response) {
  grid_map::Position requestedSubmapPosition(request->position_x, request->position_y);
  grid_map::Length requestedSubmapLength(request->length_x, request->length_y);
  RCLCPP_DEBUG(nodeHandle_->get_logger(), "Elevation submap request: Position x=%f, y=%f, Length x=%f, y=%f.", requestedSubmapPosition.x(), requestedSubmapPosition.y(),
            requestedSubmapLength(0), requestedSubmapLength(1));
  boost::recursive_mutex::scoped_lock scopedLock(map_.getFusedDataMutex());
  map_.fuseArea(requestedSubmapPosition, requestedSubmapLength);

  bool isSuccess;  
  grid_map::GridMap subMap = map_.getFusedGridMap().getSubmap(requestedSubmapPosition, requestedSubmapLength, isSuccess);
  scopedLock.unlock();

  if (request->layers.empty()) {

    response->map = *grid_map::GridMapRosConverter::toMessage(subMap);
  } else {
    std::vector<std::string> layers;
    for (const std::string& layer : request->layers) {
      layers.push_back(layer);
    }
    response->map = *grid_map::GridMapRosConverter::toMessage(subMap, layers);
  }

  RCLCPP_DEBUG(nodeHandle_->get_logger(), "Elevation submap responded with timestamp %f.", map_.getTimeOfLastFusion().seconds());
  return isSuccess;
}

bool ElevationMapping::getRawSubmapServiceCallback(std::shared_ptr<grid_map_msgs::srv::GetGridMap::Request> request,
                                                   std::shared_ptr<grid_map_msgs::srv::GetGridMap::Response> response) {
  grid_map::Position requestedSubmapPosition(request->position_x, request->position_y);
  grid_map::Length requestedSubmapLength(request->length_x, request->length_y);
  RCLCPP_DEBUG(nodeHandle_->get_logger(), "Elevation raw submap request: Position x=%f, y=%f, Length x=%f, y=%f.", requestedSubmapPosition.x(),
            requestedSubmapPosition.y(), requestedSubmapLength(0), requestedSubmapLength(1));
  boost::recursive_mutex::scoped_lock scopedLock(map_.getRawDataMutex());

  bool isSuccess;
  grid_map::GridMap subMap = map_.getRawGridMap().getSubmap(requestedSubmapPosition, requestedSubmapLength, isSuccess);
  scopedLock.unlock();

  if (request->layers.empty()) {
    response->map = *grid_map::GridMapRosConverter::toMessage(subMap);
  } else {
    std::vector<std::string> layers;
    for (const std::string& layer : request->layers) {
      layers.push_back(layer);
    }
    response->map = *grid_map::GridMapRosConverter::toMessage(subMap, layers);
  }
  return isSuccess;
}

bool ElevationMapping::disableUpdatesServiceCallback(const std::shared_ptr<rmw_request_id_t>,
                                               const std::shared_ptr<std_srvs::srv::Empty::Request>,
                                               std::shared_ptr<std_srvs::srv::Empty::Response>) {
  RCLCPP_INFO(nodeHandle_->get_logger(), "Disabling updates.");
  updatesEnabled_ = false;
  return true;
}

bool ElevationMapping::enableUpdatesServiceCallback(const std::shared_ptr<rmw_request_id_t>,
                                               const std::shared_ptr<std_srvs::srv::Empty::Request>,
                                               std::shared_ptr<std_srvs::srv::Empty::Response>) {
  RCLCPP_INFO(nodeHandle_->get_logger(), "Enabling updates.");
  updatesEnabled_ = true;
  return true;
}

bool ElevationMapping::initializeElevationMap() {
  if (initializeElevationMap_) {
    if (static_cast<elevation_mapping::InitializationMethods>(initializationMethod_) ==
        elevation_mapping::InitializationMethods::PlanarFloorInitializer) {
      geometry_msgs::msg::TransformStamped transform_msg;
      tf2::Stamped<tf2::Transform> transform;

      // Listen to transform between mapFrameId_ and targetFrameInitSubmap_ and use z value for initialization
      try {
        transform_msg = transformBuffer_->lookupTransform(mapFrameId_, targetFrameInitSubmap_, rclcpp::Time(0), rclcpp::Duration::from_seconds(5.0));
        tf2::fromMsg(transform_msg, transform);

        RCLCPP_DEBUG_STREAM(nodeHandle_->get_logger(), "Initializing with x: " << transform.getOrigin().x() << " y: " << transform.getOrigin().y()
                                                 << " z: " << transform.getOrigin().z());

        const grid_map::Position positionRobot(transform.getOrigin().x(), transform.getOrigin().y());

        // Move map before we apply the height values. This prevents unwanted behavior from intermediate move() calls in
        // updateMapLocation().
        map_.move(positionRobot);

        map_.setRawSubmapHeight(positionRobot, transform.getOrigin().z() + initSubmapHeightOffset_, lengthInXInitSubmap_,
                                lengthInYInitSubmap_, marginInitSubmap_);
        return true;
      } catch (tf2::TransformException& ex) {
        RCLCPP_DEBUG(nodeHandle_->get_logger(), "%s", ex.what());
        RCLCPP_WARN(nodeHandle_->get_logger(), "Could not initialize elevation map with constant height. (This warning can be ignored if TF tree is not available.)");
        return false;
      }
    }
  }
  return true;
}

bool ElevationMapping::clearMapServiceCallback(const std::shared_ptr<rmw_request_id_t>,
                                               const std::shared_ptr<std_srvs::srv::Empty::Request>,
                                               std::shared_ptr<std_srvs::srv::Empty::Response>) {
  RCLCPP_INFO(nodeHandle_->get_logger(), "Clearing map...");
  bool success = map_.clear();
  success &= initializeElevationMap();
  RCLCPP_INFO(nodeHandle_->get_logger(), "Map cleared.");

  return success;
}

bool ElevationMapping::maskedReplaceServiceCallback(std::shared_ptr<grid_map_msgs::srv::SetGridMap::Request> request,
                                                    std::shared_ptr<grid_map_msgs::srv::SetGridMap::Response> /*response*/) {
  RCLCPP_INFO(nodeHandle_->get_logger(), "Masked replacing of map.");
  grid_map::GridMap sourceMap;
  grid_map::GridMapRosConverter::fromMessage(request->map, sourceMap);

  // Use the supplied mask or do not use a mask
  grid_map::Matrix mask;
  if (sourceMap.exists(maskedReplaceServiceMaskLayerName_)) {
    mask = sourceMap[maskedReplaceServiceMaskLayerName_];
  } else {
    mask = Eigen::MatrixXf::Ones(sourceMap.getSize()(0), sourceMap.getSize()(1));
  }

  boost::recursive_mutex::scoped_lock scopedLockRawData(map_.getRawDataMutex());

  // Loop over all layers that should be set
  for (auto sourceLayerIterator = sourceMap.getLayers().begin(); sourceLayerIterator != sourceMap.getLayers().end();
       sourceLayerIterator++) {
    // skip "mask" layer
    if (*sourceLayerIterator == maskedReplaceServiceMaskLayerName_) {
      continue;
    }
    grid_map::Matrix& sourceLayer = sourceMap[*sourceLayerIterator];
    // Check if the layer exists in the elevation map
    if (map_.getRawGridMap().exists(*sourceLayerIterator)) {
      grid_map::Matrix& destinationLayer = map_.getRawGridMap()[*sourceLayerIterator];
      for (grid_map::GridMapIterator destinationIterator(map_.getRawGridMap()); !destinationIterator.isPastEnd(); ++destinationIterator) {
        // Use the position to find corresponding indices in source and destination
        const grid_map::Index destinationIndex(*destinationIterator);
        grid_map::Position position;
        map_.getRawGridMap().getPosition(*destinationIterator, position);

        if (!sourceMap.isInside(position)) {
          continue;
        }

        grid_map::Index sourceIndex;
        sourceMap.getIndex(position, sourceIndex);
        // If the mask allows it, set the value from source to destination
        if (!std::isnan(mask(sourceIndex(0), sourceIndex(1)))) {
          destinationLayer(destinationIndex(0), destinationIndex(1)) = sourceLayer(sourceIndex(0), sourceIndex(1));
        }
      }
    } else {
      RCLCPP_ERROR(nodeHandle_->get_logger(), "Masked replace service: Layer %s does not exist!", sourceLayerIterator->c_str());
    }
  }

  return true;
}

bool ElevationMapping::saveMapServiceCallback(std::shared_ptr<grid_map_msgs::srv::ProcessFile::Request> request,
                                              std::shared_ptr<grid_map_msgs::srv::ProcessFile::Response> response) {
  RCLCPP_INFO(nodeHandle_->get_logger(), "Saving map to file.");
  boost::recursive_mutex::scoped_lock scopedLock(map_.getFusedDataMutex());
  map_.fuseAll();
  std::string topic = std::string(nodeHandle_->get_namespace()) + "/elevation_map";
  if (!request->topic_name.empty()) {
    topic = std::string(nodeHandle_->get_namespace()) + "/" + request->topic_name;
  }
  response->success = static_cast<unsigned char>(grid_map::GridMapRosConverter::saveToBag(map_.getFusedGridMap(), request->file_path, topic));
  response->success = static_cast<unsigned char>(
      (grid_map::GridMapRosConverter::saveToBag(map_.getRawGridMap(), request->file_path + "_raw", topic + "_raw")) &&
      static_cast<bool>(response->success));
  return static_cast<bool>(response->success);
}

bool ElevationMapping::loadMapServiceCallback(std::shared_ptr<grid_map_msgs::srv::ProcessFile::Request> request,
                                              std::shared_ptr<grid_map_msgs::srv::ProcessFile::Response> response) {
  RCLCPP_WARN(nodeHandle_->get_logger(), "Loading from bag file.");
  boost::recursive_mutex::scoped_lock scopedLockFused(map_.getFusedDataMutex());
  boost::recursive_mutex::scoped_lock scopedLockRaw(map_.getRawDataMutex());

  std::string topic = nodeHandle_->get_namespace();
  if (!request->topic_name.empty()) {
    topic += "/" + request->topic_name;
  } else {
    topic += "/elevation_map";
  }

  response->success =
      static_cast<unsigned char>(grid_map::GridMapRosConverter::loadFromBag(request->file_path, topic, map_.getFusedGridMap()));
  response->success = static_cast<unsigned char>(
      grid_map::GridMapRosConverter::loadFromBag(request->file_path + "_raw", topic + "_raw", map_.getRawGridMap()) &&
      static_cast<bool>(response->success));

  // Update timestamp for visualization in ROS
  map_.setTimestamp(nodeHandle_->get_clock()->now());
  map_.postprocessAndPublishRawElevationMap();
  return static_cast<bool>(response->success);
}

// void ElevationMapping::resetMapUpdateTimer() {
//   mapUpdateTimer_->cancel();
//   rclcpp::Duration periodSinceLastUpdate = nodeHandle_->get_clock()->now() - map_.getTimeOfLastUpdate();
//   if (periodSinceLastUpdate > maxNoUpdateDuration_) {
//     periodSinceLastUpdate = rclcpp::Duration::from_seconds(0.0);
//   }
//   mapUpdateTimer_.setPeriod(maxNoUpdateDuration_ - periodSinceLastUpdate);
//   mapUpdateTimer_.start();
// }

// void ElevationMapping::stopMapUpdateTimer() {
//   mapUpdateTimer_->cancel();
// }

}  // namespace elevation_mapping
