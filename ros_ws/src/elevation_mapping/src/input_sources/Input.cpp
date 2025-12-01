/*
 *  Input.cpp
 *
 *  Created on: Oct 06, 2020
 *  Author: Magnus Gärtner
 *  Institute: ETH Zurich, ANYbotics
 */

#include "elevation_mapping/input_sources/Input.hpp"

#include "elevation_mapping/sensor_processors/LaserSensorProcessor.hpp"
#include "elevation_mapping/sensor_processors/PerfectSensorProcessor.hpp"
#include "elevation_mapping/sensor_processors/StereoSensorProcessor.hpp"
#include "elevation_mapping/sensor_processors/StructuredLightSensorProcessor.hpp"

namespace elevation_mapping {

Input::Input(std::shared_ptr<rclcpp::Node> nh) : nodeHandle_(nh) {}

bool Input::configure(std::string& inputSourceName, const std::string& sourceConfigurationName, const SensorProcessorBase::GeneralParameters& generalSensorProcessorParameters) {
  // TODO: make nicer add checkes
  // TODO: sourceConfigurationName not used
  
  Parameters parameters;

  std::string sensorProcessorType;
  
  nodeHandle_->declare_parameter(inputSourceName + ".topic","");
  nodeHandle_->declare_parameter(inputSourceName + ".queue_size",1);
  nodeHandle_->declare_parameter(inputSourceName + ".publish_on_update",true);
  nodeHandle_->declare_parameter(inputSourceName + ".sensor_processor.type","");
  nodeHandle_->declare_parameter(inputSourceName + ".type","");

  if (!nodeHandle_->get_parameter(inputSourceName + ".type", parameters.type_)){
    RCLCPP_ERROR(nodeHandle_->get_logger(), "Could not configure input source %s because no type was given.", inputSourceName.c_str());
  }

  if (!nodeHandle_->get_parameter(inputSourceName + ".topic", parameters.topic_)){
    RCLCPP_ERROR(nodeHandle_->get_logger(), "Could not configure input source %s because no topic was given.", inputSourceName.c_str());
  }

  if (!nodeHandle_->get_parameter(inputSourceName + ".queue_size", parameters.queueSize_)){
    RCLCPP_ERROR(nodeHandle_->get_logger(), "Could not configure input source %s because no queue_size was given.", inputSourceName.c_str());
  }

  if (!nodeHandle_->get_parameter(inputSourceName + ".publish_on_update", parameters.publishOnUpdate_)){
    RCLCPP_ERROR(nodeHandle_->get_logger(), "Could not configure input source %s because no publish_on_update was given.", inputSourceName.c_str());
  }

  if (!nodeHandle_->get_parameter(inputSourceName + ".sensor_processor.type", sensorProcessorType)){
    RCLCPP_ERROR(nodeHandle_->get_logger(), "Could not configure input source %s because no sensor_processor was given.", inputSourceName.c_str());
  }
  //RCLCPP_INFO(nodeHandle_->get_logger(), "Configured %s!!!!\n",parameters.publishOnUpdate_ );
  parameters.name_ = inputSourceName;
  
  parameters_.setData(parameters);
  
  

  // SensorProcessor
  if (!configureSensorProcessor(inputSourceName, sensorProcessorType, generalSensorProcessorParameters)) {
    return false;
  }

  RCLCPP_INFO(nodeHandle_->get_logger(), "Configured %s:%s @ %s (publishing_on_update: %s), using %s to process data.\n", parameters.type_.c_str(), parameters.name_.c_str(),
            rclcpp::expand_topic_or_service_name(parameters.topic_, nodeHandle_->get_name(), nodeHandle_->get_namespace()).c_str(), parameters.publishOnUpdate_ ? "true" : "false", sensorProcessorType.c_str());
  return true;
}

std::string Input::getSubscribedTopic() const {
  const Parameters parameters{parameters_.getData()};
  // FIXME: 
  return rclcpp::expand_topic_or_service_name(parameters.topic_, nodeHandle_->get_name(), nodeHandle_->get_namespace());
}

bool Input::configureSensorProcessor(std::string& inputSourceName, const std::string& sensorType,
                                     const SensorProcessorBase::GeneralParameters& generalSensorProcessorParameters) {
  
  if (sensorType == "structured_light") {
    sensorProcessor_.reset(new StructuredLightSensorProcessor(nodeHandle_, generalSensorProcessorParameters));
  } else if (sensorType == "stereo") {
    sensorProcessor_.reset(new StereoSensorProcessor(nodeHandle_, generalSensorProcessorParameters));
  } else if (sensorType == "laser") {
    sensorProcessor_.reset(new LaserSensorProcessor(nodeHandle_, generalSensorProcessorParameters));
  } else if (sensorType == "perfect") {
    sensorProcessor_.reset(new PerfectSensorProcessor(nodeHandle_, generalSensorProcessorParameters));
  } else {
    RCLCPP_ERROR(nodeHandle_->get_logger(), "The sensor type %s is not available.", sensorType.c_str());
    return false;
  }

  return sensorProcessor_->readParameters(inputSourceName);
}

}  // namespace elevation_mapping
