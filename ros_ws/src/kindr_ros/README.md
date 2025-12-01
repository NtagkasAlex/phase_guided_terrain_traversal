# Kindr ROS

Contact  : Remo Diethelm [rdiethelm ( at ) anybotics.com]

Author(s): Christian Gehring, Peter Fankhauser, Remo Diethelm

**Authors: Christian Gehring, Peter Fankhauser, Remo Diethelm<br />
Maintainer: Remo Diethelm, rdiethelm@anybotics.com<br />
Affiliation: [ANYbotics](https://www.anybotics.com/)**

This projected was initially developed at ETH Zurich (Autonomous Systems Lab & Robotic Systems Lab).

This work is conducted as part of [ANYmal Research](https://www.anymal-research.org/), a community to advance legged robotics.

The source code is released under a [BSD 3-Clause license](LICENSE).

## Overview

ROS messages and RViz plugins for kindr objects.

## Packages

### kindr_msgs (supports ROS2)

ROS messages for kindr objects.

### kindr_ros (supports ROS2)

Conversion between official ROS messages and kindr objects.

### kindr_rviz_plugins (Not yet ported)

RViz plugins to visualize kindr ROS messages.

### multi_dof_joint_trajectory_rviz_plugins (Not yet ported)

RViz plugins to visualize trajectory ROS messages.

## Usage

To build, clone this repository into your colcon workspace and type

    colcon build --packages-up-to <package_name>

To run tests type:

    colcon test --packages-select <package_name>
    colcon test-result --all --verbose
