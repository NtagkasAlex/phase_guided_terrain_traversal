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

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco.mjx._src import math
from mujoco_playground._src import collision

from go2.utility import quat_to_yaw
import go2.go2_constants as consts
from scipy.spatial.transform import Rotation
from functools import partial

def get_assets() -> Dict[str, bytes]:
  assets = {}
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "assets")

  return assets


class Go2Env(mjx_env.MjxEnv):
  """Base class for Go2 environments."""

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)
    # print(consts.ROOT_PATH)
    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=get_assets()
    )
    
    self._mj_model.opt.timestep = self._config.sim_dt

    # Modify PD gains.
    self._mj_model.dof_damping[6:] = config.Kd
    self._mj_model.actuator_gainprm[:, 0] = config.Kp
    self._mj_model.actuator_biasprm[:, 1] = -config.Kp

    # Increase offscreen framebuffer size to render at higher resolutions.
    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model)
    self._xml_path = xml_path
    self._imu_site_id = self._mj_model.site("imu").id

    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])
    self.init_feet_pos=jp.zeros((4,3))
    # Note: First joint is freejoint.
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    
    self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

    self._feet_site_id = jp.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = jp.array([self._mj_model.geom("floor").id])
    
    for i in range(self._mj_model.nbody):
      body_name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, i)  # Get body name
      if body_name and body_name.startswith("box_"):
      # print(body_name)
        
        self._floor_geom_id=jp.hstack([self._floor_geom_id,self._mj_model.body_geomadr[i]])

    self._feet_geom_id = jp.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    self._cmd_u_max = jp.array(self._config.command_config.u_max)
    self._cmd_u_min = jp.array(self._config.command_config.u_min)

    self._cmd_b = jp.array(self._config.command_config.b)
  # Sensor readings.

  def get_upvector(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, consts.UPVECTOR_SENSOR)

  def get_gravity(self, data: mjx.Data) -> jax.Array:
    return data.site_xmat[self._imu_site_id].T @ jp.array([0, 0, -1])

  def get_global_linvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.GLOBAL_LINVEL_SENSOR
    )

  def get_global_angvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.GLOBAL_ANGVEL_SENSOR
    )

  def get_local_linvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.LOCAL_LINVEL_SENSOR
    )

  def get_accelerometer(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.ACCELEROMETER_SENSOR
    )

  def get_gyro(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, consts.GYRO_SENSOR)

  def get_feet_pos(self, data: mjx.Data) -> jax.Array:
    return jp.vstack([
        mjx_env.get_sensor_data(self.mj_model, data, sensor_name)
        for sensor_name in consts.FEET_POS_SENSOR
    ])
  def get_yaw(self,data:mjx.Data):
    quat=data.qpos[3:7]
    return quat_to_yaw(quat)
  @staticmethod
  def compute_contact(data, feet_geom_ids, floor_geom_ids):
    """Check if each foot geom collides with any floor geom efficiently using JAX."""
    
    # Define the function to check collision for a single foot and a single floor
    def check_collision(foot_geom_id, floor_geom_id):
        return collision.geoms_colliding(data, foot_geom_id, floor_geom_id)
    
    # Vectorize over both feet and floor geoms (result shape: (num_feet, num_floors))
    check_collision_vmap = jax.vmap(
        jax.vmap(check_collision, in_axes=(None, 0)),  # Vectorize floor geoms for each foot geom
        in_axes=(0, None)  # Vectorize over feet geoms
    )
    
    # Compute collisions (final shape: (num_feet, num_floors))
    collision_matrix = check_collision_vmap(feet_geom_ids, floor_geom_ids)
    
    # Return a vector indicating if there is any collision for each foot (shape: (num_feet,))
    return jp.any(collision_matrix, axis=-1)
  # def raycast_sensor(self,mjx_data, pos):

  #   ray_sensor_site = jp.array([pos[0], pos[1], pos[2]])
  #   direction_vector = jp.array([0, 0, -1.])
  #   geomgroup_mask=(1,0,0,0,1,1)

  #   f_ray=partial(mjx.ray,vec=direction_vector,
  #         geomgroup=geomgroup_mask)
    
  #   z=f_ray(self.mjx_model,mjx_data,ray_sensor_site)
  #   intersection_point = ray_sensor_site + direction_vector * z[0]
    
  #   return intersection_point

  # def create_sensor_matrix(self,dx:mjx.Data,center, yaw=0.):
  #     """
  #     This is the main function used to create the grid map using the ray sensor data
  #     """

  #     R_W2H = jp.array([jp.cos(yaw), jp.sin(yaw), -jp.sin(yaw), jp.cos(yaw)])
  #     R_W2H = R_W2H.reshape((2, 2))

  #     c = int((consts.num_heightscans - 1) / 2)

  #     ref_robot = jp.array([center[0], center[1], center[2] + 0.6])

  #     idx = jp.arange(consts.num_heightscans)
  #     p, k = jp.meshgrid(idx - c, idx - c, indexing="ij")

  #     offsets = jp.stack([p * consts.dist_x, k * consts.dist_y], axis=-1)  # (N, N, 2)
  #     offsets = offsets @ R_W2H

  #     grid_positions = jp.concatenate([
  #       ref_robot[:2] + offsets,
  #       jp.full((consts.num_heightscans, consts.num_heightscans, 1), ref_robot[2])  
  #     ], axis=-1)

  #     sensor_matrix = grid_positions.at[c, c].set(ref_robot) #(N,N,3)

      
  #     get_data=jax.vmap(jax.vmap(self.raycast_sensor,in_axes=(None,0)),in_axes=(None,0))(dx,sensor_matrix)

  #     return get_data

  # Accessors.
  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model