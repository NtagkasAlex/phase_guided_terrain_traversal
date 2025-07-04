# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Defines Unitree go2 quadruped constants."""

from etils import epath

from mujoco_playground._src import mjx_env

ROOT_PATH = epath.Path("go2")
FEET_ONLY_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly.xml"
)
FEET_ONLY_FLAT_STAIRS_XML = (
    ROOT_PATH / "xmls" / "terrain_scene_mjx.xml"
)
TEST_STAIRS_XML = (
    ROOT_PATH / "xmls" / "terrain_test_mjx.xml"
)

HUGE_STAIRS_XML = (
    ROOT_PATH / "xmls" / "huge_stairs.xml"
)
FEET_ONLY_ROUGH_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_rough_terrain.xml"
)
FULL_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx.xml"

FULL_COLLISIONS_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_fullcollisions_flat_terrain.xml"
)


def task_to_xml(task_name: str) -> epath.Path:
  return {
      "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
      "stairs": FEET_ONLY_FLAT_STAIRS_XML,
      "test_stairs": TEST_STAIRS_XML,
      "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
      "huge_stairs":HUGE_STAIRS_XML,
  }[task_name]


FEET_SITES = [
    "FR_foot",
    "FL_foot",
    "RR_foot",
    "RL_foot",
]

FEET_GEOMS = [
    "FR",
    "FL",
    "RR",
    "RL",
]

FEET_POS_SENSOR = [
    "FR_pos",
    "FL_pos",
    "RR_pos",
    "RL_pos",
]

ROOT_BODY = "base"

UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

# num_heightscans=15
# num_widthscans=9

# dist_x=0.06
# dist_y=0.06
num_heightscans=13
num_widthscans=9

dist_x=0.1
dist_y=0.1
# num_heightscans=5
# num_widthscans=5

# dist_x=0.1
# dist_y=0.1

# num_heightscans=13
# num_widthscans=9

# dist_x=0.07
# dist_y=0.07