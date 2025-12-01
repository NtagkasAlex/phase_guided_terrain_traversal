/*
 * ElevationMap.cpp
 *
 *  Created on: Feb 5, 2014
 *      Author: Péter Fankhauser
 *	 Institute: ETH Zurich, ANYbotics
 */

#include <cmath>
#include <cstring>

#include <grid_map_msgs/msg/grid_map.hpp>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
#include "nav_msgs/msg/occupancy_grid.hpp"

#include "elevation_mapping/ElevationMap.hpp"
#include "elevation_mapping/ElevationMapFunctors.hpp"
#include "elevation_mapping/PointXYZRGBConfidenceRatio.hpp"
#include "elevation_mapping/WeightedEmpiricalCumulativeDistributionFunction.hpp"

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <Eigen/Dense>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include <grid_map_ros/grid_map_ros.hpp>
#include <filters/filter_chain.hpp>


namespace {
/**
 * Store an unsigned integer value in a float
 * @param input integer
 * @return A float with the bit pattern of the input integer
 */
float intAsFloat(const uint32_t input) {
  float output;
  std::memcpy(&output, &input, sizeof(uint32_t));
  return output;
}
}  // namespace

namespace elevation_mapping {

ElevationMap::ElevationMap(std::shared_ptr<rclcpp::Node> nodeHandle)
    : nodeHandle_(nodeHandle),
      rawMap_({"elevation", "variance", "horizontal_variance_x", "horizontal_variance_y", 
      "horizontal_variance_xy", "color", "time","dynamic_time", "lowest_scan_point", 
      "sensor_x_at_lowest_scan", "sensor_y_at_lowest_scan", "sensor_z_at_lowest_scan"}),
      fusedMap_({"elevation", "upper_bound", "lower_bound", "color"}),
      // FIXME: Postprocessor num threads should be same as number of filters
      postprocessorPool_(nodeHandle_->get_parameter("postprocessor_num_threads").as_int(), nodeHandle_),
      hasUnderlyingMap_(false),
      minVariance_(0.000009),
      maxVariance_(0.0009),
      mahalanobisDistanceThreshold_(1.5),
      multiHeightNoise_(0.000009),
      minHorizontalVariance_(0.0001),
      maxHorizontalVariance_(0.05),
      enableVisibilityCleanup_(true),
      enableContinuousCleanup_(false),
      visibilityCleanupDuration_(0.0),
      scanningDuration_(1.0) {
  rawMap_.setBasicLayers({"elevation", "variance"});
  fusedMap_.setBasicLayers({"elevation", "upper_bound", "lower_bound"});
  clear();

  elevationMapFusedPublisher_ = nodeHandle_->create_publisher<grid_map_msgs::msg::GridMap>("elevation_map", 1);
  // heightMapFusedPublisher_ = nodeHandle_->create_publisher<sensor_msgs::msg::PointCloud2>("elevation_heightmap", 1);



  if (!underlyingMapTopic_.empty()) {
    underlyingMapSubscriber_ = nodeHandle_->create_subscription<grid_map_msgs::msg::GridMap>(underlyingMapTopic_, 1, std::bind(&ElevationMap::underlyingMapCallback, this, std::placeholders::_1));
  }
  // TODO(max): if (enableVisibilityCleanup_) when parameter cleanup is ready.
  visibilityCleanupMapPublisher_ = nodeHandle_->create_publisher<grid_map_msgs::msg::GridMap>("visibility_cleanup_map", 1);

  initialTime_ = nodeHandle_->get_clock()->now();
}

ElevationMap::~ElevationMap() = default;

bool ElevationMap::add(const PointCloudType::Ptr pointCloud, Eigen::VectorXf& pointCloudVariances, const rclcpp::Time& timestamp,
                       const Eigen::Affine3d& transformationSensorToMap) {
  if (static_cast<unsigned int>(pointCloud->size()) != static_cast<unsigned int>(pointCloudVariances.size())) {
    RCLCPP_ERROR(nodeHandle_->get_logger(), "ElevationMap::add: Size of point cloud (%i) and variances (%i) do not agree.", (int)pointCloud->size(),
              (int)pointCloudVariances.size());
    return false;
  }
  // RCLCPP_INFO(nodeHandle_->get_logger(), "Point cloud variances: %f, %f, %f", pointCloudVariances(0), pointCloudVariances(1), pointCloudVariances(2)) ;

  // Initialization for time calculation.
  // RCLCPP_INFO(nodeHandle_->get_logger(), "ElevationMap::add: Initializing map.");
  const auto methodStartTime = std::chrono::system_clock::now();
  const rclcpp::Time currentTime = nodeHandle_->get_clock()->now();
  const float currentTimeSecondsPattern{intAsFloat(static_cast<uint32_t>(static_cast<uint64_t>(currentTime.seconds())))};
  boost::recursive_mutex::scoped_lock scopedLockForRawData(rawMapMutex_);

  // Update initial time if it is not initialized.
  if (initialTime_.seconds() == 0) {
    initialTime_ = timestamp;
  }
  const float scanTimeSinceInitialization = (timestamp - initialTime_).seconds();

  // Store references for efficient interaction.
  // RCLCPP_INFO(nodeHandle_->get_logger(), "ElevationMap::add: Storing references.");
  auto& elevationLayer = rawMap_["elevation"];
  auto& varianceLayer = rawMap_["variance"];
  auto& horizontalVarianceXLayer = rawMap_["horizontal_variance_x"];
  auto& horizontalVarianceYLayer = rawMap_["horizontal_variance_y"];
  auto& horizontalVarianceXYLayer = rawMap_["horizontal_variance_xy"];
  auto& colorLayer = rawMap_["color"];
  auto& timeLayer = rawMap_["time"];
  auto& dynamicTimeLayer = rawMap_["dynamic_time"];
  auto& lowestScanPointLayer = rawMap_["lowest_scan_point"];
  auto& sensorXatLowestScanLayer = rawMap_["sensor_x_at_lowest_scan"];
  auto& sensorYatLowestScanLayer = rawMap_["sensor_y_at_lowest_scan"];
  auto& sensorZatLowestScanLayer = rawMap_["sensor_z_at_lowest_scan"];

  std::vector<Eigen::Ref<const grid_map::Matrix>> basicLayers_;
  for (const std::string& layer : rawMap_.getBasicLayers()) {
    basicLayers_.push_back(rawMap_.get(layer));
  }

  for (unsigned int i = 0; i < pointCloud->size(); ++i) {
    auto& point = pointCloud->points[i];
    grid_map::Index index;
    grid_map::Position position(point.x, point.y);  // NOLINT(cppcoreguidelines-pro-type-union-access)
    // RCLCPP_INFO(nodeHandle_->get_logger(), "Position in grid map: %f, %f", position.x(), position.y());
    if (!rawMap_.getIndex(position, index)) {
      continue;  // Skip this point if it does not lie within the elevation map.
    }

    auto& elevation = elevationLayer(index(0), index(1));
    auto& variance = varianceLayer(index(0), index(1));
    auto& horizontalVarianceX = horizontalVarianceXLayer(index(0), index(1));
    auto& horizontalVarianceY = horizontalVarianceYLayer(index(0), index(1));
    auto& horizontalVarianceXY = horizontalVarianceXYLayer(index(0), index(1));
    auto& color = colorLayer(index(0), index(1));
    auto& time = timeLayer(index(0), index(1));
    auto& dynamicTime = dynamicTimeLayer(index(0), index(1));
    auto& lowestScanPoint = lowestScanPointLayer(index(0), index(1));
    auto& sensorXatLowestScan = sensorXatLowestScanLayer(index(0), index(1));
    auto& sensorYatLowestScan = sensorYatLowestScanLayer(index(0), index(1));
    auto& sensorZatLowestScan = sensorZatLowestScanLayer(index(0), index(1));

    const float& pointVariance = 1e-11 * pointCloudVariances(i);
    bool isValid = std::all_of(basicLayers_.begin(), basicLayers_.end(),
                               [&](Eigen::Ref<const grid_map::Matrix> layer) { 
                                return std::isfinite(layer(index(0), index(1))); });
    if (!isValid) {
      // No prior information in elevation map, use measurement.
      // RCLCPP_INFO(nodeHandle_->get_logger(), "No prior information in elevation map, using measurement.");
      elevation = point.z;  // NOLINT(cppcoreguidelines-pro-type-union-access)
      // RCLCPP_INFO(nodeHandle_->get_logger(), "Elevation is %f", elevation);
      variance = pointVariance;
      horizontalVarianceX = minHorizontalVariance_;
      horizontalVarianceY = minHorizontalVariance_;
      horizontalVarianceXY = 0.0;
      grid_map::colorVectorToValue(point.getRGBVector3i(), color);
      continue;
    }

    // RCLCPP_INFO(nodeHandle_->get_logger(), "Elevation: %f, Point Z: %f, Variance: %f", elevation, point.z, variance);
    // Deal with multiple heights in one cell.
    const double mahalanobisDistance = fabs(point.z - elevation) / sqrt(variance);  // NOLINT(cppcoreguidelines-pro-type-union-access)
    // RCLCPP_INFO(nodeHandle_->get_logger(), "Mahalanobis Distance: %f", mahalanobisDistance);
    if (mahalanobisDistance > mahalanobisDistanceThreshold_) {
      // RCLCPP_INFO(nodeHandle_->get_logger(), "Mahalanobis distance exceeds threshold.");
      if (scanTimeSinceInitialization - time <= scanningDuration_ &&
          elevation > point.z) {  // NOLINT(cppcoreguidelines-pro-type-union-access)
          // RCLCPP_INFO(nodeHandle_->get_logger(), "Ignoring point, lower than existing elevation and within scanning duration.");
        // Ignore point if measurement is from the same point cloud (time comparison) and
        // if measurement is lower then the elevation in the map.
      } else if (scanTimeSinceInitialization - time <= scanningDuration_) {
        // RCLCPP_INFO(nodeHandle_->get_logger(), "Point is higher, updating elevation and variance.");
        // If point is higher.
        elevation = increaseHeightAlpha_ * elevation +
                    (1.0 - increaseHeightAlpha_) * point.z;  // NOLINT(cppcoreguidelines-pro-type-union-access)
        variance = increaseHeightAlpha_ * variance + (1.0 - increaseHeightAlpha_) * pointVariance;
        
      } else {
        // RCLCPP_INFO(nodeHandle_->get_logger(), "Increasing variance due to multi-height noise.");
        variance += multiHeightNoise_;
      }
        continue;
    }

    // Store lowest points from scan for visibility checking.
    const float pointHeightPlusUncertainty =  point.z + 3.0 * sqrt(pointVariance);  // 3 sigma. // NOLINT(cppcoreguidelines-pro-type-union-access)
    if (std::isnan(lowestScanPoint) || pointHeightPlusUncertainty < lowestScanPoint) {
      // RCLCPP_INFO(nodeHandle_->get_logger(), "Updating lowest scan point.");
      lowestScanPoint = pointHeightPlusUncertainty;
      const grid_map::Position3 sensorTranslation(transformationSensorToMap.translation());
      sensorXatLowestScan = sensorTranslation.x();
      sensorYatLowestScan = sensorTranslation.y();
      sensorZatLowestScan = sensorTranslation.z();
    }

    // RCLCPP_INFO(nodeHandle_->get_logger(), "Fusing measurement with elevation map data.");
    // Fuse measurement with elevation map data. aka kalman filter
    elevation =
        (variance * point.z + pointVariance * elevation) / (variance + pointVariance);  // NOLINT(cppcoreguidelines-pro-type-union-access)
    variance = (pointVariance * variance) / (pointVariance + variance);
    // TODO(max): Add color fusion.
    grid_map::colorVectorToValue(point.getRGBVector3i(), color);
    time = scanTimeSinceInitialization;
    dynamicTime = currentTimeSecondsPattern;

    // Horizontal variances are reset.
    horizontalVarianceX = minHorizontalVariance_;
    horizontalVarianceY = minHorizontalVariance_;
    horizontalVarianceXY = 0.0;
    // RCLCPP_INFO(nodeHandle_->get_logger(), "Updated map cell: Elevation = %f, Variance = %f", elevation, variance);
  }

  //std::stringstream ss;
  //ss << sensorXatLowestScanLayer.format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]"));
  //RCLCPP_INFO(nodeHandle_->get_logger(), "robotPoseCovariance: \n%s", ss.str().c_str() );
  
  clean();
  rawMap_.setTimestamp(timestamp.nanoseconds());  // Point cloud stores time in microseconds.

  const std::chrono::duration<double> duration  = std::chrono::system_clock::now() - methodStartTime;
  // RCLCPP_INFO(nodeHandle_->get_logger(), "Raw map has been updated with a new point cloud in %f s.", duration.count());
  return true;
}

bool ElevationMap::update(const grid_map::Matrix& varianceUpdate, const grid_map::Matrix& horizontalVarianceUpdateX,
                          const grid_map::Matrix& horizontalVarianceUpdateY, const grid_map::Matrix& horizontalVarianceUpdateXY,
                          const rclcpp::Time& time) {
  boost::recursive_mutex::scoped_lock scopedLock(rawMapMutex_);

  const auto& size = rawMap_.getSize();

  if (!((grid_map::Index(varianceUpdate.rows(), varianceUpdate.cols()) == size).all() &&
        (grid_map::Index(horizontalVarianceUpdateX.rows(), horizontalVarianceUpdateX.cols()) == size).all() &&
        (grid_map::Index(horizontalVarianceUpdateY.rows(), horizontalVarianceUpdateY.cols()) == size).all() &&
        (grid_map::Index(horizontalVarianceUpdateXY.rows(), horizontalVarianceUpdateXY.cols()) == size).all())) {
    RCLCPP_ERROR(nodeHandle_->get_logger(), "The size of the update matrices does not match.");
    return false;
  }

  rawMap_.get("variance") += varianceUpdate;
  rawMap_.get("horizontal_variance_x") += horizontalVarianceUpdateX;
  rawMap_.get("horizontal_variance_y") += horizontalVarianceUpdateY;
  rawMap_.get("horizontal_variance_xy") += horizontalVarianceUpdateXY;
  clean();
  rawMap_.setTimestamp(time.nanoseconds());

  return true;
}

bool ElevationMap::fuse(const grid_map::Index& topLeftIndex, const grid_map::Index& size) {
  RCLCPP_DEBUG(nodeHandle_->get_logger(), "Fusing elevation map...");

  // Nothing to do.
  if ((size == 0).any()) {
    return false;
  }

  // Initializations.
  const auto methodStartTime = std::chrono::system_clock::now();  

  // Copy raw elevation map data for safe multi-threading.
  boost::recursive_mutex::scoped_lock scopedLockForRawData(rawMapMutex_);
  auto rawMapCopy = rawMap_;
  scopedLockForRawData.unlock();

  boost::recursive_mutex::scoped_lock scopedLock(fusedMapMutex_);

  // More initializations.
  const double halfResolution = fusedMap_.getResolution() / 2.0;
  const float minimalWeight = std::numeric_limits<float>::epsilon() * static_cast<float>(2.0);
  // Conservative cell inclusion for ellipse iterator.
  const double ellipseExtension = M_SQRT2 * fusedMap_.getResolution();

  // Check if there is the need to reset out-dated data.
  if (fusedMap_.getTimestamp() != rawMapCopy.getTimestamp()) {
    resetFusedData();
  }

  // Align fused map with raw map.
  if (rawMapCopy.getPosition() != fusedMap_.getPosition()) {
    fusedMap_.move(rawMapCopy.getPosition());
  }

  // For each cell in requested area.
  for (grid_map::SubmapIterator areaIterator(rawMapCopy, topLeftIndex, size); !areaIterator.isPastEnd(); ++areaIterator) {
    // Check if fusion for this cell has already been done earlier.
    if (fusedMap_.isValid(*areaIterator)) {
      continue;
    }

    if (!rawMapCopy.isValid(*areaIterator)) {
      // This is an empty cell (hole in the map).
      // TODO(max):
      continue;
    }

    // Get size of error ellipse.
    const float& sigmaXsquare = rawMapCopy.at("horizontal_variance_x", *areaIterator);
    const float& sigmaYsquare = rawMapCopy.at("horizontal_variance_y", *areaIterator);
    const float& sigmaXYsquare = rawMapCopy.at("horizontal_variance_xy", *areaIterator);

    Eigen::Matrix2d covarianceMatrix;
    covarianceMatrix << sigmaXsquare, sigmaXYsquare, sigmaXYsquare, sigmaYsquare;
    // 95.45% confidence ellipse which is 2.486-sigma for 2 dof problem.
    // http://www.reid.ai/2012/09/chi-squared-distribution-table-with.html
    const double uncertaintyFactor = 2.486;  // sqrt(6.18)
    Eigen::EigenSolver<Eigen::Matrix2d> solver(covarianceMatrix);
    Eigen::Array2d eigenvalues(solver.eigenvalues().real().cwiseAbs());

    Eigen::Array2d::Index maxEigenvalueIndex;
    eigenvalues.maxCoeff(&maxEigenvalueIndex);
    Eigen::Array2d::Index minEigenvalueIndex;
    maxEigenvalueIndex == Eigen::Array2d::Index(0) ? minEigenvalueIndex = 1 : minEigenvalueIndex = 0;
    const grid_map::Length ellipseLength =
        2.0 * uncertaintyFactor * grid_map::Length(eigenvalues(maxEigenvalueIndex), eigenvalues(minEigenvalueIndex)).sqrt() +
        ellipseExtension;
    const double ellipseRotation(
        atan2(solver.eigenvectors().col(maxEigenvalueIndex).real()(1), solver.eigenvectors().col(maxEigenvalueIndex).real()(0)));

    // Requested length and position (center) of submap in map.
    grid_map::Position requestedSubmapPosition;
    rawMapCopy.getPosition(*areaIterator, requestedSubmapPosition);
    grid_map::EllipseIterator ellipseIterator(rawMapCopy, requestedSubmapPosition, ellipseLength, ellipseRotation);

    // Prepare data fusion.
    Eigen::ArrayXf means, weights;
    const unsigned int maxNumberOfCellsToFuse = ellipseIterator.getSubmapSize().prod();
    means.resize(maxNumberOfCellsToFuse);
    weights.resize(maxNumberOfCellsToFuse);
    WeightedEmpiricalCumulativeDistributionFunction<float> lowerBoundDistribution;
    WeightedEmpiricalCumulativeDistributionFunction<float> upperBoundDistribution;

    float maxStandardDeviation = sqrt(eigenvalues(maxEigenvalueIndex));
    float minStandardDeviation = sqrt(eigenvalues(minEigenvalueIndex));
    Eigen::Rotation2Dd rotationMatrix(ellipseRotation);
    std::string maxEigenvalueLayer, minEigenvalueLayer;
    if (maxEigenvalueIndex == 0) {
      maxEigenvalueLayer = "horizontal_variance_x";
      minEigenvalueLayer = "horizontal_variance_y";
    } else {
      maxEigenvalueLayer = "horizontal_variance_y";
      minEigenvalueLayer = "horizontal_variance_x";
    }

    // For each cell in error ellipse.
    size_t i = 0;
    for (; !ellipseIterator.isPastEnd(); ++ellipseIterator) {
      if (!rawMapCopy.isValid(*ellipseIterator)) {
        // Empty cell in submap (cannot be center cell because we checked above).
        continue;
      }

      means[i] = rawMapCopy.at("elevation", *ellipseIterator);

      // Compute weight from probability.
      grid_map::Position absolutePosition;
      rawMapCopy.getPosition(*ellipseIterator, absolutePosition);
      Eigen::Vector2d distanceToCenter = (rotationMatrix * (absolutePosition - requestedSubmapPosition)).cwiseAbs();

      float probability1 = cumulativeDistributionFunction(distanceToCenter.x() + halfResolution, 0.0, maxStandardDeviation) -
                           cumulativeDistributionFunction(distanceToCenter.x() - halfResolution, 0.0, maxStandardDeviation);
      float probability2 = cumulativeDistributionFunction(distanceToCenter.y() + halfResolution, 0.0, minStandardDeviation) -
                           cumulativeDistributionFunction(distanceToCenter.y() - halfResolution, 0.0, minStandardDeviation);

      const float weight = std::max(minimalWeight, probability1 * probability2);
      weights[i] = weight;
      const float standardDeviation = sqrt(rawMapCopy.at("variance", *ellipseIterator));
      lowerBoundDistribution.add(means[i] - 2.0 * standardDeviation, weight);
      upperBoundDistribution.add(means[i] + 2.0 * standardDeviation, weight);

      i++;
    }

    if (i == 0) {
      // Nothing to fuse.
      fusedMap_.at("elevation", *areaIterator) = rawMapCopy.at("elevation", *areaIterator);
      fusedMap_.at("lower_bound", *areaIterator) =
          rawMapCopy.at("elevation", *areaIterator) - 2.0 * sqrt(rawMapCopy.at("variance", *areaIterator));
      fusedMap_.at("upper_bound", *areaIterator) =
          rawMapCopy.at("elevation", *areaIterator) + 2.0 * sqrt(rawMapCopy.at("variance", *areaIterator));
      fusedMap_.at("color", *areaIterator) = rawMapCopy.at("color", *areaIterator);
      continue;
    }

    // Fuse.
    means.conservativeResize(i);
    weights.conservativeResize(i);

    float mean = (weights * means).sum() / weights.sum();

    if (!std::isfinite(mean)) {
      RCLCPP_ERROR(nodeHandle_->get_logger(), "Something went wrong when fusing the map: Mean = %f", mean);
      continue;
    }

    // Add to fused map.
    fusedMap_.at("elevation", *areaIterator) = mean;
    lowerBoundDistribution.compute();
    upperBoundDistribution.compute();
    fusedMap_.at("lower_bound", *areaIterator) = lowerBoundDistribution.quantile(0.01);  // TODO(max):
    fusedMap_.at("upper_bound", *areaIterator) = upperBoundDistribution.quantile(0.99);  // TODO(max):
    // TODO(max): Add fusion of colors.
    fusedMap_.at("color", *areaIterator) = rawMapCopy.at("color", *areaIterator);
  }

  fusedMap_.setTimestamp(rawMapCopy.getTimestamp());

  const std::chrono::duration<double> duration = std::chrono::system_clock::now() - methodStartTime;
  RCLCPP_DEBUG(nodeHandle_->get_logger(), "Elevation map has been fused in %f s.", duration.count());

  return true;
}




void ElevationMap::visibilityCleanup(const rclcpp::Time& updatedTime) {
  // Get current time to compute calculation time.
  const auto methodStartTime = std::chrono::system_clock::now();
  const double timeSinceInitialization = (updatedTime - initialTime_).seconds();

  // Copy raw elevation map data for safe multi-threading.
  boost::recursive_mutex::scoped_lock scopedLockForVisibilityCleanupData(visibilityCleanupMapMutex_);
  boost::recursive_mutex::scoped_lock scopedLockForRawData(rawMapMutex_);
  visibilityCleanupMap_ = rawMap_;
  rawMap_.clear("lowest_scan_point");
  rawMap_.clear("sensor_x_at_lowest_scan");
  rawMap_.clear("sensor_y_at_lowest_scan");
  rawMap_.clear("sensor_z_at_lowest_scan");
  scopedLockForRawData.unlock();
  visibilityCleanupMap_.add("max_height");

  // Create max. height layer with ray tracing.
  for (grid_map::GridMapIterator iterator(visibilityCleanupMap_); !iterator.isPastEnd(); ++iterator) {
    if (!visibilityCleanupMap_.isValid(*iterator)) {
      continue;
    }
    const auto& lowestScanPoint = visibilityCleanupMap_.at("lowest_scan_point", *iterator);
    const auto& sensorXatLowestScan = visibilityCleanupMap_.at("sensor_x_at_lowest_scan", *iterator);
    const auto& sensorYatLowestScan = visibilityCleanupMap_.at("sensor_y_at_lowest_scan", *iterator);
    const auto& sensorZatLowestScan = visibilityCleanupMap_.at("sensor_z_at_lowest_scan", *iterator);
    if (std::isnan(lowestScanPoint)) {
      continue;
    }
    grid_map::Index indexAtSensor;
    if (!visibilityCleanupMap_.getIndex(grid_map::Position(sensorXatLowestScan, sensorYatLowestScan), indexAtSensor)) {
      continue;
    }
    grid_map::Position point;
    visibilityCleanupMap_.getPosition(*iterator, point);
    float pointDiffX = point.x() - sensorXatLowestScan;
    float pointDiffY = point.y() - sensorYatLowestScan;
    float distanceToPoint = sqrt(pointDiffX * pointDiffX + pointDiffY * pointDiffY);
    if (distanceToPoint > 0.0) {
      for (grid_map::LineIterator iterator(visibilityCleanupMap_, indexAtSensor, *iterator); !iterator.isPastEnd(); ++iterator) {
        grid_map::Position cellPosition;
        visibilityCleanupMap_.getPosition(*iterator, cellPosition);
        const float cellDiffX = cellPosition.x() - sensorXatLowestScan;
        const float cellDiffY = cellPosition.y() - sensorYatLowestScan;
        const float distanceToCell = distanceToPoint - sqrt(cellDiffX * cellDiffX + cellDiffY * cellDiffY);
        const float maxHeightPoint = lowestScanPoint + (sensorZatLowestScan - lowestScanPoint) / distanceToPoint * distanceToCell;
        auto& cellMaxHeight = visibilityCleanupMap_.at("max_height", *iterator);
        if (std::isnan(cellMaxHeight) || cellMaxHeight > maxHeightPoint) {
          cellMaxHeight = maxHeightPoint;
        }
      }
    }
  }

  // Vector of indices that will be removed.
  std::vector<grid_map::Position> cellPositionsToRemove;
  for (grid_map::GridMapIterator iterator(visibilityCleanupMap_); !iterator.isPastEnd(); ++iterator) {
    if (!visibilityCleanupMap_.isValid(*iterator)) {
      continue;
    }
    const auto& time = visibilityCleanupMap_.at("time", *iterator);
    if (timeSinceInitialization - time > scanningDuration_) {
      // Only remove cells that have not been updated during the last scan duration.
      // This prevents a.o. removal of overhanging objects.
      const auto& elevation = visibilityCleanupMap_.at("elevation", *iterator);
      const auto& variance = visibilityCleanupMap_.at("variance", *iterator);
      const auto& maxHeight = visibilityCleanupMap_.at("max_height", *iterator);
      if (!std::isnan(maxHeight) && elevation - 3.0 * sqrt(variance) > maxHeight) {
        grid_map::Position position;
        visibilityCleanupMap_.getPosition(*iterator, position);
        cellPositionsToRemove.push_back(position);
      }
    }
  }

  // Remove points in current raw map.
  scopedLockForRawData.lock();
  for (const auto& cellPosition : cellPositionsToRemove) {
    grid_map::Index index;
    if (!rawMap_.getIndex(cellPosition, index)) {
      continue;
    }
    if (rawMap_.isValid(index)) {
      rawMap_.at("elevation", index) = NAN;
      rawMap_.at("dynamic_time", index) = 0.0f;
    }
  }
  scopedLockForRawData.unlock();

  // Publish visibility cleanup map for debugging.
  publishVisibilityCleanupMap();

  const std::chrono::duration<double> duration = std::chrono::system_clock::now() - methodStartTime;
  RCLCPP_DEBUG(nodeHandle_->get_logger(), "Visibility cleanup has been performed in %f s (%d points).", duration.count(), (int)cellPositionsToRemove.size());
  if (duration.count() > visibilityCleanupDuration_) {
    RCLCPP_WARN(nodeHandle_->get_logger(), "Visibility cleanup duration is too high (current rate is %f).", 1.0 / duration.count());
  }
}

void ElevationMap::move(const Eigen::Vector2d& position) {
  boost::recursive_mutex::scoped_lock scopedLockForRawData(rawMapMutex_);
  std::vector<grid_map::BufferRegion> newRegions;

  if (rawMap_.move(position, newRegions)) {
    RCLCPP_DEBUG(nodeHandle_->get_logger(), "Elevation map has been moved to position (%f, %f).", rawMap_.getPosition().x(), rawMap_.getPosition().y());

    // The "dynamic_time" layer is meant to be interpreted as integer values, therefore nan:s need to be zeroed.
    grid_map::Matrix& dynTime{rawMap_.get("dynamic_time")};
    dynTime = dynTime.array().isNaN().select(grid_map::Matrix::Scalar(0.0f), dynTime.array());

    if (hasUnderlyingMap_) {
      rawMap_.addDataFrom(underlyingMap_, false, false, true);
    }
  }
}






void ElevationMap::setRawSubmapHeight(const grid_map::Position& initPosition, float mapHeight, double lengthInXSubmap,
                                      double lengthInYSubmap, double margin) {
  // Set a submap area (lengthInYSubmap + margin, lengthInXSubmap + margin) with a constant height (mapHeight).
  boost::recursive_mutex::scoped_lock scopedLockForRawData(rawMapMutex_);

  // Calculate submap iterator start index.
  const grid_map::Position topLeftPosition(initPosition(0) + lengthInXSubmap / 2, initPosition(1) + lengthInYSubmap / 2);
  grid_map::Index submapTopLeftIndex;
  rawMap_.getIndex(topLeftPosition, submapTopLeftIndex);

  // Calculate submap area.
  const double resolution = rawMap_.getResolution();
  const int lengthInXSubmapI = static_cast<int>(lengthInXSubmap / resolution + 2 * margin);
  const int lengthInYSubmapI = static_cast<int>(lengthInYSubmap / resolution + 2 * margin);
  const Eigen::Array2i submapBufferSize(lengthInYSubmapI, lengthInXSubmapI);

  // Iterate through submap and fill height values.
  grid_map::Matrix& elevationData = rawMap_["elevation"];
  grid_map::Matrix& varianceData = rawMap_["variance"];
  for (grid_map::SubmapIterator iterator(rawMap_, submapTopLeftIndex, submapBufferSize); !iterator.isPastEnd(); ++iterator) {
    const grid_map::Index index(*iterator);
    elevationData(index(0), index(1)) = mapHeight;
    varianceData(index(0), index(1)) = 0.0;
  }
}


bool ElevationMap::postprocessAndPublishRawElevationMap() {
  if (!hasRawMapSubscribers()) {
    return false;
  }
  boost::recursive_mutex::scoped_lock scopedLock(rawMapMutex_);
  grid_map::GridMap rawMapCopy = rawMap_; 
  scopedLock.unlock();
  return postprocessorPool_.runTask(rawMapCopy);
}
bool ElevationMap::applyInpaintingFilter(const grid_map::GridMap& input_map, grid_map::GridMap& output_map) {
  // Instantiate filter with template parameter
  // grid_map::InpaintFilter<grid_map::GridMap> inpaintFilter;

  // // Set parameters manually using a map or directly
  // inpaintFilter.setParameters({
  //   {"input_layer", "elevation"},
  //   {"output_layer", "elevation_inpainted"},
  //   {"radius", 0.05},
  //   {"method", std::string("TELEA")}  // or "NS"
  // });

  // // Or explicitly set method enum if needed
  // // inpaintFilter.setMethod(grid_map::InpaintFilter<grid_map::GridMap>::Method::TELEA);

  // if (!inpaintFilter.update(input_map, output_map)) {
  //   RCLCPP_ERROR(rclcpp::get_logger("applyInpaintingFilter"), "Failed to apply inpainting filter.");
  //   return false;
  // }
  return true;
}
bool ElevationMap::publishFusedElevationMap() {
  if (!hasFusedMapSubscribers()) {
    return false;
  }
  boost::recursive_mutex::scoped_lock scopedLock(fusedMapMutex_);
  grid_map::GridMap fusedMapCopy = fusedMap_;
  scopedLock.unlock();
  fusedMapCopy.add("uncertainty_range", fusedMapCopy.get("upper_bound") - fusedMapCopy.get("lower_bound"));
  std::unique_ptr<grid_map_msgs::msg::GridMap> message;
  message = grid_map::GridMapRosConverter::toMessage(fusedMapCopy);


  // nodeHandle_->declare_parameter("filters_names", std::string("filters"));

  // nodeHandle_->get_parameter("filters_names", filterChainParametersName_);
  

  // // Setup filter chain.
  // if (filterChain_.configure(
  //     filterChainParametersName_, nodeHandle_->get_node_logging_interface(),
  //     nodeHandle_->get_node_parameters_interface()))
  // {
  //   RCLCPP_INFO(nodeHandle_->get_logger(), "Filter chain configured.");
  // } else {
  //   RCLCPP_ERROR(nodeHandle_->get_logger(), "Could not configure the filter chain!");
  //   rclcpp::shutdown();
  //   return false;
  // }

  
  // grid_map::GridMap outputMap;
  // filterChain_.update(fusedMapCopy, outputMap);
  // // if (!filterChain_.update(fusedMapCopy, outputMap)) {
  // //   RCLCPP_ERROR(nodeHandle_->get_logger(), "Could not update the grid map filter chain!");
  // //   return false;
  // // }


  // std::unique_ptr<grid_map_msgs::msg::GridMap> message;
  // message = grid_map::GridMapRosConverter::toMessage(outputMap);

  elevationMapFusedPublisher_->publish(std::move(message));
  // heightMapFusedPublisher_->publish(heightmap_msg);
  // RCLCPP_DEBUG(nodeHandle_->get_logger(), "Elevation map (fused) has been published.");
  return true;
}

bool ElevationMap::publishVisibilityCleanupMap() {
  if (visibilityCleanupMapPublisher_->get_subscription_count() < 1) {
    return false;
  }
  boost::recursive_mutex::scoped_lock scopedLock(visibilityCleanupMapMutex_);
  grid_map::GridMap visibilityCleanupMapCopy = visibilityCleanupMap_;
  scopedLock.unlock();
  visibilityCleanupMapCopy.erase("elevation");
  visibilityCleanupMapCopy.erase("variance");
  visibilityCleanupMapCopy.erase("horizontal_variance_x");
  visibilityCleanupMapCopy.erase("horizontal_variance_y");
  visibilityCleanupMapCopy.erase("horizontal_variance_xy");
  visibilityCleanupMapCopy.erase("color");
  visibilityCleanupMapCopy.erase("time");
  std::unique_ptr<grid_map_msgs::msg::GridMap> message;
  message = grid_map::GridMapRosConverter::toMessage(visibilityCleanupMapCopy);
  visibilityCleanupMapPublisher_->publish(std::move(message));
  RCLCPP_DEBUG(nodeHandle_->get_logger(), "Visibility cleanup map has been published.");
  return true;
}

void ElevationMap::underlyingMapCallback(const grid_map_msgs::msg::GridMap::SharedPtr underlyingMap) {
  RCLCPP_INFO(nodeHandle_->get_logger(), "Updating underlying map.");
  grid_map::GridMapRosConverter::fromMessage(*underlyingMap, underlyingMap_);
  if (underlyingMap_.getFrameId() != rawMap_.getFrameId()) {
    RCLCPP_ERROR_STREAM(nodeHandle_->get_logger(), "The underlying map does not have the same map frame ('" << underlyingMap_.getFrameId() << "') as the elevation map ('"
                                                                              << rawMap_.getFrameId() << "').");
    return;
  }
  if (!underlyingMap_.exists("elevation")) {
    RCLCPP_ERROR_STREAM(nodeHandle_->get_logger(), "The underlying map does not have an 'elevation' layer.");
    return;
  }
  if (!underlyingMap_.exists("variance")) {
    underlyingMap_.add("variance", minVariance_);
  }
  if (!underlyingMap_.exists("horizontal_variance_x")) {
    underlyingMap_.add("horizontal_variance_x", minHorizontalVariance_);
  }
  if (!underlyingMap_.exists("horizontal_variance_y")) {
    underlyingMap_.add("horizontal_variance_y", minHorizontalVariance_);
  }
  if (!underlyingMap_.exists("color")) {
    underlyingMap_.add("color", 0.0);
  }
  underlyingMap_.setBasicLayers(rawMap_.getBasicLayers());
  hasUnderlyingMap_ = true;
  rawMap_.addDataFrom(underlyingMap_, false, false, true);
}







bool ElevationMap::clear() {
  // Lock raw and fused map object in different scopes to prevent deadlock.
  {
    boost::recursive_mutex::scoped_lock scopedLockForRawData(rawMapMutex_);
    rawMap_.clearAll();
    rawMap_.resetTimestamp();
    rawMap_.get("dynamic_time").setZero();
  }
  {
    boost::recursive_mutex::scoped_lock scopedLockForFusedData(fusedMapMutex_);
    fusedMap_.clearAll();
    fusedMap_.resetTimestamp();
  }
  return true;
} 

void ElevationMap::setGeometry(const grid_map::Length& length, const double& resolution, const grid_map::Position& position) {
  boost::recursive_mutex::scoped_lock scopedLockForRawData(rawMapMutex_);
  boost::recursive_mutex::scoped_lock scopedLockForFusedData(fusedMapMutex_);
  rawMap_.setGeometry(length, resolution, position);
  fusedMap_.setGeometry(length, resolution, position);
  RCLCPP_INFO_STREAM(nodeHandle_->get_logger(), "Elevation map grid resized to " << rawMap_.getSize()(0) << " rows and " << rawMap_.getSize()(1) << " columns.");
}

float ElevationMap::cumulativeDistributionFunction(float x, float mean, float standardDeviation) {
  return 0.5 * erfc(-(x - mean) / (standardDeviation * sqrt(2.0)));
}

bool ElevationMap::hasFusedMapSubscribers() const {
  return elevationMapFusedPublisher_->get_subscription_count() >= 1;
}

  // return (elevationMapFusedPublisher_->get_subscription_count() >= 1 || heightMapFusedPublisher_->get_subscription_count() >= 1 );
// }

//sets frame id for both maps.
void ElevationMap::setFrameId(const std::string& frameId) {
  rawMap_.setFrameId(frameId);
  fusedMap_.setFrameId(frameId);
}

//gives time in nanoseconds (of type RCL_TIME) to the maps.
void ElevationMap::setTimestamp(rclcpp::Time timestamp) {
  rawMap_.setTimestamp(timestamp.nanoseconds());
  fusedMap_.setTimestamp(timestamp.nanoseconds());
}

//gets frame id from map raw.
const std::string& ElevationMap::getFrameId() {
  return rawMap_.getFrameId();
}

bool ElevationMap::hasRawMapSubscribers() const {
  return postprocessorPool_.pipelineHasSubscribers();
}

bool ElevationMap::clean() {
  boost::recursive_mutex::scoped_lock scopedLockForRawData(rawMapMutex_);
  rawMap_.get("variance") = rawMap_.get("variance").unaryExpr(VarianceClampOperator<float>(minVariance_, maxVariance_));
  rawMap_.get("horizontal_variance_x") =
      rawMap_.get("horizontal_variance_x").unaryExpr(VarianceClampOperator<float>(minHorizontalVariance_, maxHorizontalVariance_));
  rawMap_.get("horizontal_variance_y") =
      rawMap_.get("horizontal_variance_y").unaryExpr(VarianceClampOperator<float>(minHorizontalVariance_, maxHorizontalVariance_));
  return true;
}

void ElevationMap::resetFusedData() {
  boost::recursive_mutex::scoped_lock scopedLockForFusedData(fusedMapMutex_);
  fusedMap_.clearAll();
  fusedMap_.resetTimestamp();
}

bool ElevationMap::fuseAll() {
  RCLCPP_DEBUG(nodeHandle_->get_logger(), "Requested to fuse entire elevation map.");
  boost::recursive_mutex::scoped_lock scopedLock(fusedMapMutex_);
  return fuse(grid_map::Index(0, 0), fusedMap_.getSize());
}

bool ElevationMap::fuseArea(const Eigen::Vector2d& position, const Eigen::Array2d& length) {
  RCLCPP_DEBUG(nodeHandle_->get_logger(), "Requested to fuse an area of the elevation map with center at (%f, %f) and side lengths (%f, %f)", position[0], position[1],
            length[0], length[1]);

  grid_map::Index topLeftIndex;
  grid_map::Index submapBufferSize;

  // These parameters are not used in this function.
  grid_map::Position submapPosition;
  grid_map::Length submapLength;
  grid_map::Index requestedIndexInSubmap;

  boost::recursive_mutex::scoped_lock scopedLock(fusedMapMutex_);

  grid_map::getSubmapInformation(topLeftIndex, submapBufferSize, submapPosition, submapLength, requestedIndexInSubmap, position, length,
                                 rawMap_.getLength(), rawMap_.getPosition(), rawMap_.getResolution(), rawMap_.getSize(),
                                 rawMap_.getStartIndex());
  // change topLeftIndex and buffersize for constant size...
  return fuse(topLeftIndex, submapBufferSize);
}

grid_map::GridMap& ElevationMap::getRawGridMap() {
  return rawMap_;
}

void ElevationMap::setRawGridMap(const grid_map::GridMap& map) {
  boost::recursive_mutex::scoped_lock scopedLockForRawData(rawMapMutex_);
  rawMap_ = map;
}

grid_map::GridMap& ElevationMap::getFusedGridMap() {
  return fusedMap_;
}

void ElevationMap::setFusedGridMap(const grid_map::GridMap& map) {
  boost::recursive_mutex::scoped_lock scopedLockForFusedData(fusedMapMutex_);
  fusedMap_ = map;
}

rclcpp::Time ElevationMap::getTimeOfLastUpdate() {
  return rclcpp::Time(rawMap_.getTimestamp(), RCL_ROS_TIME);
}

rclcpp::Time ElevationMap::getTimeOfLastFusion() {
  boost::recursive_mutex::scoped_lock scopedLock(fusedMapMutex_);
  return rclcpp::Time(fusedMap_.getTimestamp(), RCL_ROS_TIME);
}

const kindr::HomTransformQuatD& ElevationMap::getPose() {
  return pose_;
}

bool ElevationMap::getPosition3dInRobotParentFrame(const Eigen::Array2i& index, kindr::Position3D& position) {
  kindr::Position3D positionInGridFrame;
  if (!rawMap_.getPosition3("elevation", index, positionInGridFrame.vector())) {
    return false;
  }
  position = pose_.transform(positionInGridFrame);
  return true;
}

boost::recursive_mutex& ElevationMap::getFusedDataMutex() {
  return fusedMapMutex_;
}

boost::recursive_mutex& ElevationMap::getRawDataMutex() {
  return rawMapMutex_;
}


void ElevationMap::getHeightMap(const Eigen::Vector3d& position, double yaw) {
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

            float elevation = std::numeric_limits<float>::quiet_NaN();  // Default if index is invalid

            if (this->getFusedGridMap().getIndex(gridPos, index)) {
                elevation = this->getFusedGridMap().at("elevation", index);
            }

            size_t point_offset = i * heightmap_msg.row_step + j * heightmap_msg.point_step;

            float x = static_cast<float>(query_pos.x());
            float y = static_cast<float>(query_pos.y());
            float z = elevation;

            std::memcpy(&heightmap_msg.data[point_offset + 0], &x, sizeof(float));
            std::memcpy(&heightmap_msg.data[point_offset + sizeof(float)], &y, sizeof(float));
            std::memcpy(&heightmap_msg.data[point_offset + 2 * sizeof(float)], &z, sizeof(float));
        }
    }
}
  // Now 'yaw' contains the rotation around the Z-axis in radians
  // std::cout << "Yaw (radians): " << yaw << std::endl;
  // std::cout << "Yaw (degrees): " << yaw * 180.0 / M_PI << std::endl;

//   float x=1.5;
//   float y=0.0;
//   grid_map::Position position(x, y);
//   grid_map::Index index;

// // Try to get the index from the position
//   if (map_.getFusedGridMap().getIndex(position, index)) {
  // RCLCPP_INFO(nodeHandle_->get_logger(), "Size: (%d, %d)", num_heightscans,num_widthscans);
  // RCLCPP_INFO(nodeHandle_->get_logger(), "Position in grid map");

//   } else {
//     RCLCPP_WARN(nodeHandle_->get_logger(), "Position is outside the grid map bounds.");
//   }
  


  // RCLCPP_INFO(nodeHandle_->get_logger(),
  //           "Transform origin: x = %.3f, y = %.3f",
  //           transform.getOrigin().x(),
  //           transform.getOrigin().y());
  // grid_map::Index index(0, 0);
  // RCLCPP_INFO(nodeHandle_->get_logger(), "Elevation: %.2f", map_.getFusedGridMap().at("elevation", index));
  







}  // namespace elevation_mapping
