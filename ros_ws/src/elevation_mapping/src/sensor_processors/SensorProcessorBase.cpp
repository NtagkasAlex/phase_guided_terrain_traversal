/*
 * SensorProcessorBase.cpp
 *
 *  Created on: Jun 6, 2014
 *      Author: Péter Fankhauser, Hannes Keller
 *   Institute: ETH Zurich, ANYbotics
 */

#include "elevation_mapping/sensor_processors/SensorProcessorBase.hpp"

// ROS
#include <geometry_msgs/msg/transform_stamped.hpp>

// PCL
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/pcl_base.h>

// TF
#include <tf2_eigen/tf2_eigen.hpp>

// STL
#include <cmath>
#include <limits>
#include <vector>

#include "elevation_mapping/PointXYZRGBConfidenceRatio.hpp"

namespace elevation_mapping {

SensorProcessorBase::SensorProcessorBase(std::shared_ptr<rclcpp::Node>& nodeHandle, const GeneralParameters& generalConfig)
    : nodeHandle_(nodeHandle),
      // transformListener_(std::make_shared<tf2_ros::TransformListener>(*transformBuffer_)),
      // ignorePointsUpperThreshold_(std::numeric_limits<double>::infinity()),
      // ignorePointsLowerThreshold_(-std::numeric_limits<double>::infinity()),
      //applyVoxelGridFilter_(false),
      firstTfAvailable_(false) {
  transformBuffer_ = std::make_shared<tf2_ros::Buffer>(nodeHandle_->get_clock());
  transformListener_ = std::make_shared<tf2_ros::TransformListener>(*transformBuffer_);

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
  transformationSensorToMap_.setIdentity();
  generalParameters_ = generalConfig;
  // RCLCPP_DEBUG(nodeHandle_->get_logger(),
  //     "Sensor processor general parameters are:"
  //     "\n\t- robot_base_frame_id: %s"
  //     "\n\t- map_frame_id: %s",
  //     generalConfig.robotBaseFrameId_.c_str(), generalConfig.mapFrameId_.c_str());
}

SensorProcessorBase::~SensorProcessorBase() = default;

bool SensorProcessorBase::readParameters(std::string& inputSourceName) {
  Parameters parameters;

  // TODO: Add parameter validation
  nodeHandle_->declare_parameter(inputSourceName + ".sensor_processor.ignore_points_above", rclcpp::ParameterValue(std::numeric_limits<double>::infinity()));
  nodeHandle_->declare_parameter(inputSourceName + ".sensor_processor.ignore_points_below", rclcpp::ParameterValue(-std::numeric_limits<double>::infinity()));
  nodeHandle_->declare_parameter(inputSourceName + ".sensor_processor.apply_voxelgrid_filter", rclcpp::ParameterValue(false));
  nodeHandle_->declare_parameter(inputSourceName + ".sensor_processor.voxelgrid_filter_size", rclcpp::ParameterValue(0.0));

  nodeHandle_->declare_parameter(inputSourceName + ".sensor_processor.ignore_points_inside_min_x", 0.0);
  nodeHandle_->declare_parameter(inputSourceName + ".sensor_processor.ignore_points_inside_max_x", 0.0);
  nodeHandle_->declare_parameter(inputSourceName + ".sensor_processor.ignore_points_inside_min_y", 0.0);
  nodeHandle_->declare_parameter(inputSourceName + ".sensor_processor.ignore_points_inside_max_y", 0.0);
  nodeHandle_->declare_parameter(inputSourceName + ".sensor_processor.ignore_points_inside_min_z", 0.0);
  nodeHandle_->declare_parameter(inputSourceName + ".sensor_processor.ignore_points_inside_max_z", 0.0);

  nodeHandle_->get_parameter(inputSourceName + ".sensor_processor.ignore_points_above", parameters.ignorePointsUpperThreshold_);
  nodeHandle_->get_parameter(inputSourceName + ".sensor_processor.ignore_points_below", parameters.ignorePointsLowerThreshold_);

  nodeHandle_->get_parameter(inputSourceName + ".sensor_processor.ignore_points_inside_min_x", parameters.ignorePointsInsideMinX_);
  nodeHandle_->get_parameter(inputSourceName + ".sensor_processor.ignore_points_inside_max_x", parameters.ignorePointsInsideMaxX_);
  nodeHandle_->get_parameter(inputSourceName + ".sensor_processor.ignore_points_inside_min_y", parameters.ignorePointsInsideMinY_);
  nodeHandle_->get_parameter(inputSourceName + ".sensor_processor.ignore_points_inside_max_y", parameters.ignorePointsInsideMaxY_);
  nodeHandle_->get_parameter(inputSourceName + ".sensor_processor.ignore_points_inside_min_z", parameters.ignorePointsInsideMinZ_);
  nodeHandle_->get_parameter(inputSourceName + ".sensor_processor.ignore_points_inside_max_z", parameters.ignorePointsInsideMaxZ_);

  nodeHandle_->get_parameter(inputSourceName + ".sensor_processor.apply_voxelgrid_filter", parameters.applyVoxelGridFilter_);
  nodeHandle_->get_parameter(inputSourceName + ".sensor_processor.voxelgrid_filter_size", sensorParameters_["voxelgrid_filter_size"]);

  parameters_.setData(parameters);
  return true;
}

bool SensorProcessorBase::process(const PointCloudType::ConstPtr pointCloudInput, const Eigen::Matrix<double, 6, 6>& robotPoseCovariance,
                                  const PointCloudType::Ptr pointCloudMapFrame, Eigen::VectorXf& variances, std::string sensorFrame) {
  sensorFrameId_ = sensorFrame;
  // RCLCPP_DEBUG(rclcpp::get_logger("sensor_processor"), "Sensor Processor processing for frame %s", sensorFrameId_.c_str());

  // Update transformation at timestamp of pointcloud   
  rclcpp::Time timeStamp = rclcpp::Time(1000 * pointCloudInput->header.stamp);
  if (!updateTransformations(timeStamp)) {
    return false;
  }

  // Transform into sensor frame.
  PointCloudType::Ptr pointCloudSensorFrame(new PointCloudType);
  // FIXME: transformPointCloud seems to be useless since it transfroms from pointcloud frame to the same frame
  //       sensorFrame is inputpointcloud frame
  *pointCloudSensorFrame = *pointCloudInput;
  // transformPointCloud(pointCloudInput, pointCloudSensorFrame, sensorFrameId_);

  // Remove Nans (optional voxel grid filter)
  filterPointCloud(pointCloudSensorFrame);

  // Specific filtering per sensor type
  filterPointCloudSensorType(pointCloudSensorFrame);
  // Remove outside limits in map frame
  if (!transformPointCloud(pointCloudSensorFrame, pointCloudMapFrame, generalParameters_.mapFrameId_)) {
    return false;
  }

  std::vector<PointCloudType::Ptr> pointClouds({pointCloudMapFrame, pointCloudSensorFrame});
  // FIXME: Removing of points should be done in the sensor frame
  removePointsOutsideLimits(pointCloudMapFrame, pointClouds);

  // Compute variances
  return computeVariances(pointCloudSensorFrame, robotPoseCovariance, variances);
}

bool SensorProcessorBase::updateTransformations(const rclcpp::Time& timeStamp) {
  try {

    geometry_msgs::msg::TransformStamped transformGeom;
    // FIXME: lookupTransform sometimes takes long
    transformGeom = transformBuffer_->lookupTransform(generalParameters_.mapFrameId_, sensorFrameId_, timeStamp,
                                                      rclcpp::Duration::from_seconds(1.0));
    transformationSensorToMap_ = tf2::transformToEigen(transformGeom);

    transformGeom = transformBuffer_->lookupTransform(generalParameters_.robotBaseFrameId_, sensorFrameId_, timeStamp,
                                                      rclcpp::Duration::from_seconds(1.0));  // TODO(max): Why wrong direction?
    Eigen::Quaterniond rotationQuaternion;
    tf2::fromMsg(transformGeom.transform.rotation, rotationQuaternion);
    rotationBaseToSensor_.setMatrix(rotationQuaternion.toRotationMatrix());
    Eigen::Vector3d translationVector;
    tf2::fromMsg(transformGeom.transform.translation, translationVector);
    translationBaseToSensorInBaseFrame_.toImplementation() = translationVector;
  
    transformGeom = transformBuffer_->lookupTransform(generalParameters_.mapFrameId_, generalParameters_.robotBaseFrameId_,
                                                    timeStamp, rclcpp::Duration::from_seconds(1.0));  // TODO(max): Why wrong direction?
    tf2::fromMsg(transformGeom.transform.rotation, rotationQuaternion);
    rotationMapToBase_.setMatrix(rotationQuaternion.toRotationMatrix());
    tf2::fromMsg(transformGeom.transform.translation, translationVector);
    translationMapToBaseInMapFrame_.toImplementation() = translationVector;

    if (!firstTfAvailable_) {
      firstTfAvailable_ = true;
    }

    return true;
  } catch (tf2::TransformException& ex) {
    if (!firstTfAvailable_) {
      return false;
    }
    // RCLCPP_ERROR(nodeHandle_->get_logger(), "%s", ex.what());
    return false;
  }
}

bool SensorProcessorBase::transformPointCloud(PointCloudType::ConstPtr pointCloud, PointCloudType::Ptr pointCloudTransformed,
                                              const std::string& targetFrame) {
  rclcpp::Time timeStamp = rclcpp::Time(1000 * pointCloud->header.stamp);  
  const std::string inputFrameId(pointCloud->header.frame_id);

  try {
    geometry_msgs::msg::TransformStamped transformGeom;
    transformGeom = transformBuffer_->lookupTransform(targetFrame, inputFrameId, timeStamp, rclcpp::Duration(0,1000));  // FIXME: missing 0.001 retry duration
    Eigen::Affine3d transform = tf2::transformToEigen(transformGeom);
    pcl::transformPointCloud(*pointCloud, *pointCloudTransformed, transform.cast<float>());
    pointCloudTransformed->header.frame_id = targetFrame;
    rclcpp::Clock clock;

    // FIXME: get_logger() causes SEGFAULT
    //RCLCPP_DEBUG_THROTTLE(nodeHandle_->get_logger(), clock, 5, "Point cloud transformed to frame %s for time stamp %f.", targetFrame.c_str(),
    //                  pointCloudTransformed->header.stamp / 1000.0);
  } catch (tf2::TransformException& ex) {
    // RCLCPP_ERROR(nodeHandle_->get_logger(), "%s", ex.what());
    return false;
  }

  return true;
}

void SensorProcessorBase::removePointsOutsideLimits(PointCloudType::ConstPtr reference, std::vector<PointCloudType::Ptr>& pointClouds) {
  const Parameters parameters{parameters_.getData()};
  //TODO: Update condition to include ignore inside
  if (!std::isfinite(parameters.ignorePointsLowerThreshold_) && !std::isfinite(parameters.ignorePointsUpperThreshold_)) {
    return;
  }
  // FIXME: get_logger segfault
  //RCLCPP_DEBUG(nodeHandle_->get_logger(), "Limiting point cloud to the height interval of [%f, %f] relative to the robot base.", ignorePointsLowerThreshold_,
  //          ignorePointsUpperThreshold_);

  pcl::PassThrough<pcl::PointXYZRGBConfidenceRatio> passThroughFilter(true);
  passThroughFilter.setInputCloud(reference);
  passThroughFilter.setFilterFieldName("z");  // TODO(max): Should this be configurable?
  double relativeLowerThreshold = translationMapToBaseInMapFrame_.z() + parameters.ignorePointsLowerThreshold_;
  double relativeUpperThreshold = translationMapToBaseInMapFrame_.z() + parameters.ignorePointsUpperThreshold_;
  passThroughFilter.setFilterLimits(relativeLowerThreshold, relativeUpperThreshold);
  pcl::IndicesPtr insideIndeces(new std::vector<int>);
  passThroughFilter.filter(*insideIndeces);

  for (auto& pointCloud : pointClouds) {
    pcl::ExtractIndices<pcl::PointXYZRGBConfidenceRatio> extractIndicesFilter;
    extractIndicesFilter.setInputCloud(pointCloud);
    extractIndicesFilter.setIndices(insideIndeces);
    PointCloudType tempPointCloud;
    extractIndicesFilter.filter(tempPointCloud);
    pointCloud->swap(tempPointCloud);
  }
  // RCLCPP_DEBUG(rclcpp::get_logger("sensor_processor"), "removePointsOutsideLimits() pass reduced point cloud to %i points.", (int)pointClouds[0]->size());

  // TODO: It would be beneficial to remove points in robot_frame, not map_frame 
  pcl::CropBox<pcl::PointXYZRGBConfidenceRatio> cropBoxFilter(true); // true-returns indices
  cropBoxFilter.setInputCloud(pointClouds[0]);
  cropBoxFilter.setNegative(true); // remove points inside the box
  cropBoxFilter.setMin(Eigen::Vector4f(parameters.ignorePointsInsideMinX_ + translationMapToBaseInMapFrame_.x(),
                                       parameters.ignorePointsInsideMinY_ + translationMapToBaseInMapFrame_.y(),
                                       parameters.ignorePointsInsideMinZ_ + translationMapToBaseInMapFrame_.z(), 1.0f));
  cropBoxFilter.setMax(Eigen::Vector4f(parameters.ignorePointsInsideMaxX_ + translationMapToBaseInMapFrame_.x(),
                                       parameters.ignorePointsInsideMaxY_ + translationMapToBaseInMapFrame_.y(),
                                       parameters.ignorePointsInsideMaxZ_ + translationMapToBaseInMapFrame_.z(), 1.0f));
  pcl::IndicesPtr insideIndecesCropBox(new std::vector<int>);
  cropBoxFilter.filter(*insideIndecesCropBox);  

  for (auto& pointCloud : pointClouds) {
    pcl::ExtractIndices<pcl::PointXYZRGBConfidenceRatio> extractIndicesFilter;
    extractIndicesFilter.setInputCloud(pointCloud);
    extractIndicesFilter.setIndices(insideIndecesCropBox);
    PointCloudType tempPointCloud;
    extractIndicesFilter.filter(tempPointCloud);
    pointCloud->swap(tempPointCloud);
  }
  // RCLCPP_DEBUG(rclcpp::get_logger("sensor_processor"), "removePointsOutsideLimits() box reduced point cloud to %i points.", (int)pointClouds[0]->size());
}

bool SensorProcessorBase::filterPointCloud(const PointCloudType::Ptr pointCloud) {
  const Parameters parameters{parameters_.getData()};

  PointCloudType tempPointCloud;
  
  // Remove nan points.
  std::vector<int> indices;
  if (!pointCloud->is_dense) {
    pcl::removeNaNFromPointCloud(*pointCloud, tempPointCloud, indices);
    tempPointCloud.is_dense = true;
    pointCloud->swap(tempPointCloud);
  }

  // Reduce points using VoxelGrid filter.
  if (parameters.applyVoxelGridFilter_) {
    pcl::VoxelGrid<pcl::PointXYZRGBConfidenceRatio> voxelGridFilter;
    voxelGridFilter.setInputCloud(pointCloud);
    double filter_size = sensorParameters_.at("voxelgrid_filter_size");
    voxelGridFilter.setLeafSize(filter_size, filter_size, filter_size);
    voxelGridFilter.filter(tempPointCloud);
    pointCloud->swap(tempPointCloud);
  }
  rclcpp::Clock clock;
  // FIXME: get_logger causes SEGFAULT
  //RCLCPP_DEBUG_THROTTLE(nodeHandle_->get_logger(), clock, 2, "cleanPointCloud() reduced point cloud to %i points.", static_cast<int>(pointCloud->size()));
  return true;
}

bool SensorProcessorBase::filterPointCloudSensorType(const PointCloudType::Ptr /*pointCloud*/) {
  return true;
}

} /* namespace elevation_mapping */
