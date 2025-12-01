/*
 * MedianFillFilter.cpp
 *
 *  Created on: September 7, 2020
 *      Author: Magnus Gärtner
 *   Institute: ETH Zurich, ANYbotics
 */

#include "median_fill_filter/MedianFillFilter.hpp"

#include <cmath>
#include <rclcpp/rclcpp.hpp>

// Grid Map
#include <grid_map_core/grid_map_core.hpp>
#include "grid_map_cv/utilities.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>


namespace grid_map {
template<typename T>
MedianFillFilter<T>::MedianFillFilter()
    : fillHoleRadius_(0.05), existingValueRadius_(0.0), filterExistingValues_(false), numErodeDilationIterations_(4), debug_(false) {}

template<typename T>
MedianFillFilter<T>::~MedianFillFilter()
{
  
}

template<typename T>
bool MedianFillFilter<T>::configure() {

  ParameterReader param_reader(this->param_prefix_, this->params_interface_);
  if (!param_reader.get(std::string("fill_hole_radius"), fillHoleRadius_)) {
    RCLCPP_ERROR(this->logging_interface_->get_logger(),"Median filter did not find parameter fill_hole_radius.");
    return false;
  }

  if (fillHoleRadius_ < 0.0) {
    RCLCPP_ERROR(this->logging_interface_->get_logger(),"fill_hole_radius must be greater than zero.");
    return false;
  }

  RCLCPP_DEBUG(this->logging_interface_->get_logger(),"fill_hole_radius = %f.", fillHoleRadius_);

  if (!param_reader.get(std::string("filter_existing_values"), filterExistingValues_)) {
    RCLCPP_INFO(this->logging_interface_->get_logger(),"Median filter did not find parameter filter_existing_values. Not filtering existing values.");
    filterExistingValues_ = false;
  }

  RCLCPP_DEBUG(this->logging_interface_->get_logger(),"Filter_existing_values = %s.", filterExistingValues_ ? "true" : "false");

  if (!param_reader.get(std::string("input_layer"), inputLayer_)) {
    RCLCPP_ERROR(this->logging_interface_->get_logger(),"Median filter did not find parameter `input_layer`.");
    return false;
  }

  if (!param_reader.get(std::string("num_erode_dilation_iterations"), numErodeDilationIterations_)) {
    RCLCPP_ERROR(this->logging_interface_->get_logger(),"Median filter did not find parameter `num_erode_dilation_iterations`.");
    return false;
  }

  if (filterExistingValues_) {
    if (!param_reader.get(std::string("existing_value_radius"), existingValueRadius_)) {
      RCLCPP_ERROR(this->logging_interface_->get_logger(),"Median filter did not find parameter existing_value_radius.");
      return false;
    }

    if (existingValueRadius_ < 0.0) {
      RCLCPP_ERROR(this->logging_interface_->get_logger(),"existing_value_radius must be greater than zero.");
      return false;
    }

    RCLCPP_DEBUG(this->logging_interface_->get_logger(),"existing_value_radius = %f.", existingValueRadius_);
  }

  RCLCPP_DEBUG(this->logging_interface_->get_logger(),"Median input layer is = %s.", inputLayer_.c_str());

  if (!param_reader.get(std::string("output_layer"), outputLayer_)) {
    RCLCPP_ERROR(this->logging_interface_->get_logger(),"Median filter did not find parameter `output_layer`.");
    return false;
  }

  RCLCPP_DEBUG(this->logging_interface_->get_logger(),"Median output layer = %s.", outputLayer_.c_str());

  if (!param_reader.get(std::string("fill_mask_layer"), fillMaskLayer_)) {
    RCLCPP_ERROR(this->logging_interface_->get_logger(),"Median filter did not find parameter `fill_mask_layer`.");
    return false;
  }

  RCLCPP_DEBUG(this->logging_interface_->get_logger(),"Median fill mask layer = %s.", fillMaskLayer_.c_str());

  if (!param_reader.get(std::string("debug"), debug_)) {
    RCLCPP_INFO(this->logging_interface_->get_logger(),"Median filter did not find parameter debug. Disabling debug output.");
    debug_ = false;
  }

  RCLCPP_DEBUG(this->logging_interface_->get_logger(),"Debug mode= %s.", debug_ ? "true" : "false");

  if (debug_ && !param_reader.get(std::string("debug_infill_mask_layer"), debugInfillMaskLayer_)) {
    RCLCPP_ERROR(this->logging_interface_->get_logger(),"Median filter did not find parameter `debug_infill_mask_layer`.");
    return false;
  }

  RCLCPP_DEBUG(this->logging_interface_->get_logger(),"Median debug infill mask layer = %s.", debugInfillMaskLayer_.c_str());

  return true;
}
template<typename T>
bool MedianFillFilter<T>::update(const T& mapIn, T& mapOut) {
  // Copy input map and add new layer to it.
  mapOut = mapIn;
  if (!mapOut.exists(outputLayer_)) {
    mapOut.add(outputLayer_);
  }

  mapOut.convertToDefaultStartIndex();

  // Avoid hash map lookups afterwards. I.e, get data matrices as references.
  grid_map::Matrix inputMap{mapOut[inputLayer_]};  // copy by value to do filtering first.
  grid_map::Matrix& outputMap{mapOut[outputLayer_]};

  // Check if mask is already computed from a previous iteration.
  Eigen::MatrixXf shouldFill;
  if (std::find(mapOut.getLayers().begin(), mapOut.getLayers().end(), fillMaskLayer_) == mapOut.getLayers().end()) {
    shouldFill = computeAndAddFillMask(inputMap, mapOut);
  } else {  // The mask already exists, retrieve it from a previous iteration.
    shouldFill = mapOut[fillMaskLayer_];
  }

  const size_t radiusInPixels{static_cast<size_t>(fillHoleRadius_ / mapIn.getResolution())};
  const size_t existingValueRadiusInPixels{static_cast<size_t>(existingValueRadius_ / mapIn.getResolution())};
  const grid_map::Index& bufferSize{mapOut.getSize()};
  unsigned int numNans{0u};
  // Iterate through the entire GridMap and update NaN values with the median.
  for (GridMapIterator iterator(mapOut); !iterator.isPastEnd(); ++iterator) {
    const grid_map::Index index(*iterator);
    const auto& inputValue{inputMap(index(0), index(1))};
    const float& shouldFillThisCell{shouldFill(index(0), index(1))};
    auto& outputValue{outputMap(index(0), index(1))};
    if (!std::isfinite(inputValue) && (shouldFillThisCell != 0.0f)) {  // Fill the NaN input value with the median.
      outputValue = getMedian(inputMap, index, radiusInPixels, bufferSize);
      numNans++;
    } else if (filterExistingValues_ && (shouldFillThisCell != 0.0f)) {  // Value is already finite. Optionally add some filtering.
      outputValue = getMedian(inputMap, index, existingValueRadiusInPixels, bufferSize);
    } else {  // Dont do any filtering, just take the input value.
      outputValue = inputValue;
    }
  }
  // ROS_DEBUG_STREAM("Median fill filter " << this->getName() << " observed " << numNans << " Nans in input layer!");
  // By removing all basic layers the selected layer will always be visualized, otherwise isValid will also check for the basic layers
  // and hide infilled values where the corresponding basic layers are still NAN.
  mapOut.setBasicLayers({});
  return true;
}
template<typename T>
float MedianFillFilter<T>::getMedian(Eigen::Ref<const grid_map::Matrix> inputMap, const grid_map::Index& centerIndex, const size_t radiusInPixels,
                                  const Size bufferSize) {
  // Bound the median window to the GridMap boundaries. Calculate the neighbour patch.
  grid_map::Index topLeftIndex{centerIndex - Index(radiusInPixels, radiusInPixels)};
  grid_map::Index bottomRightIndex{centerIndex + Index(radiusInPixels, radiusInPixels)};
  boundIndexToRange(topLeftIndex, bufferSize);
  boundIndexToRange(bottomRightIndex, bufferSize);
  const Index neighbourPatchSize{bottomRightIndex - topLeftIndex + Index{1, 1}};

  // Extract local neighbourhood.
  const auto& neighbourhood{inputMap.block(topLeftIndex(0), topLeftIndex(1), neighbourPatchSize(0), neighbourPatchSize(1))};

  const size_t cols{static_cast<size_t>(neighbourhood.cols())};

  std::vector<float> cellValues;
  cellValues.reserve(neighbourhood.rows() * neighbourhood.cols());

  for (Eigen::Index row = 0; row < neighbourhood.rows(); ++row) {
    const auto& currentRow{neighbourhood.row(row)};
    for (size_t col = 0; col < cols; ++col) {
      const float& cellValue{currentRow[col]};
      if (std::isfinite(cellValue)) {  // Calculate the median of the finite neighbours.
        cellValues.emplace_back(cellValue);
      }
    }
  }

  if (cellValues.empty()) {
    return std::numeric_limits<float>::quiet_NaN();
  } else {  // Compute the median of the finite values in the neighbourhood.
    std::nth_element(cellValues.begin(), cellValues.begin() + cellValues.size() / 2, cellValues.end());
    return cellValues[cellValues.size() / 2];
  }
}
template<typename T>
Eigen::MatrixXf MedianFillFilter<T>::computeAndAddFillMask(const Eigen::MatrixXf& inputMap, T& mapOut) {
  Eigen::MatrixXf shouldFill;
  // Precompute mask of valid height values
  using MaskMatrix = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;
  const MaskMatrix isValid{inputMap.array().unaryExpr([&](float v) { return std::isfinite(v); })};

  // Remove sparse valid values and fill holes.
  cv::Mat isValidCV;
  cv::eigen2cv(isValid, isValidCV);
  cv::Mat_<bool> isValidOutlierRemoved{cleanedMask(isValidCV)};
  cv::Mat shouldFillCV{fillHoles(isValidOutlierRemoved, numErodeDilationIterations_)};

  // Outlier removed mask to eigen.
  if (debug_) {
    addCvMatAsLayer(mapOut, isValidOutlierRemoved, debugInfillMaskLayer_);
  }

  // Convert to eigen and add to the map.
  cv::cv2eigen(shouldFillCV, shouldFill);
  mapOut.add(fillMaskLayer_, shouldFill);

  return shouldFill;
}
template<typename T>
cv::Mat_<bool> MedianFillFilter<T>::cleanedMask(const cv::Mat_<bool>& inputMask) {
  auto element{cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(1, 1))};

  cv::Mat_<bool> cleanedInputMask(inputMask.size(), false);

  // Erode then dilate to remove sparse points
  cv::dilate(inputMask, cleanedInputMask, element);
  cv::erode(cleanedInputMask, cleanedInputMask, element);
  cv::erode(inputMask, cleanedInputMask, element);
  cv::dilate(cleanedInputMask, cleanedInputMask, element);

  return cleanedInputMask;
}
template<typename T>
cv::Mat_<bool> MedianFillFilter<T>::fillHoles(const cv::Mat_<bool>& isValidMask, const size_t numDilationClosingIterations) {
  auto element{cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(1, 1))};
  cv::Mat_<bool> holesFilledMask(isValidMask.size(), false);
  // Remove holes in the mask by morphological closing.
  cv::dilate(isValidMask, holesFilledMask, element);
  for (size_t iteration = 1; iteration < numDilationClosingIterations; iteration++) {
    cv::dilate(holesFilledMask, holesFilledMask, element);
  }
  for (size_t iteration = 0; iteration < numDilationClosingIterations; iteration++) {
    cv::erode(holesFilledMask, holesFilledMask, element);
  }

  return holesFilledMask;
}
template<typename T>
void MedianFillFilter<T>::addCvMatAsLayer(T& gridMap, const cv::Mat& cvLayer, const std::string& layerName) {
  Eigen::MatrixXf tmpEigenMatrix;
  cv::cv2eigen(cvLayer, tmpEigenMatrix);
  gridMap.add(layerName, tmpEigenMatrix);
}

}  // namespace grid_map

PLUGINLIB_EXPORT_CLASS(grid_map::MedianFillFilter<grid_map::GridMap>, filters::FilterBase<grid_map::GridMap>)
