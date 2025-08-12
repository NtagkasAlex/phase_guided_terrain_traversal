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
"""Joystick task for go2."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
import go2.base as go2_base
import go2.go2_constants as consts
import go2.gait as gait
from go2.heightmap import create_sensor_matrix
import go2.joystick_base as joystick_base
from go2.configs import default_config

class Joystick(joystick_base.Joystick_Base):
  """Track a joystick command."""
  def __init__( 
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        task=task,
        config=config,
        config_overrides=config_overrides,
    )

 
  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)
    # state = self._reset_if_outside_bounds(state)
    oscilator_pose=gait.joint_trajectory(state.info["phase"], self._config.reward_config.swing_height,self._config.reward_config.base_feet_distance)

    motor_targets = oscilator_pose + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    # contact = jp.array([
    #     collision.geoms_colliding(data, geom_id, self._floor_geom_id)
    #     for geom_id in self._feet_geom_id
    # ])

    contact = self.compute_contact(data,self._feet_geom_id, self._floor_geom_id)
    # print(contact)
    
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = self.get_feet_pos(data)
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    # heightscan=self.create_sensor_matrix(data,data.qpos[:3],self.get_yaw(data))
    heightscan=create_sensor_matrix(self.mjx_model,data,data.qpos[:3],self.get_yaw(data))
    # jax.debug.print("{}",heightscan[...,2])
    n = (heightscan.shape[0] - 1) // 2  # This gives us the value of 'n' based on the shape of heightscan
    # Extracting the four regions of the heightscan matrix
    top_right = heightscan[:n, n+1:,2]          # Top-right corner
    top_left = heightscan[:n, :n,2]             # Top-left corner
    back_right = heightscan[n+1:, n+1:,2]       # Back-right corner
    back_left = heightscan[n+1:, :n,2]         # Back-left corner

    H_max_value = jp.array([
      jp.max(top_right),   
      jp.max(top_left),    
      jp.max(back_right),  
      jp.max(back_left)    
    ]) 
    H_min_value = jp.array([
      jp.min(top_right),   
      jp.min(top_left),    
      jp.min(back_right),  
      jp.min(back_left)    
    ]) 
    state.info["heightscan"]=heightscan
    state.info["H_max"]=H_max_value
    state.info["H_min"]=H_min_value

    obs = self._get_obs(data, state.info)

    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
    # reward = sum(rewards.values()) * self.dt

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["step"] += 1
    phase_unwrapped=state.info["phase"]+state.info["phase_dt"]
    state.info["phase"]=jp.fmod(phase_unwrapped,2*jp.pi)
    state.info["steps_until_next_cmd"] -= 1
    state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
    state.info["command"] = jp.where(
        state.info["steps_until_next_cmd"] <= 0,
        self.sample_command(key1, state.info["command"]),
        state.info["command"],
    )
    state.info["steps_until_next_cmd"] = jp.where(
        done | (state.info["steps_until_next_cmd"] <= 0),
        jp.round(jax.random.exponential(key2) * 5.0 / self.dt).astype(jp.int32),
        state.info["steps_until_next_cmd"],
    )
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_upvector(data)[-1] < 0.0
    
    return fall_termination

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> Dict[str, jax.Array]:
    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = self.get_gravity(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )
    # cos = jp.cos(info["phase"])
    # sin = jp.sin(info["phase"])
    # phase = jp.concatenate([cos, sin])

    # gyro = self.get_gyro(data)
    # gravity = self.get_gravity(data)
    # joint_angles = data.qpos[7:]
    # joint_vel = data.qvel[6:]
    # linvel = self.get_local_linvel(data)
    
    cos = jp.cos(info["phase"])
    sin = jp.sin(info["phase"])
    phase = jp.concatenate([cos, sin])

    # noisy_heightscan = (
    #     info["heightscan"]
    #     + (2 * jax.random.uniform(noise_rng, shape=info["heightscan"].shape) - 1)
    #     * self._config.noise_config.level
    #     * self._config.noise_config.scales.heightscan
    # )       
    # z_values = noisy_heightscan[..., 2].ravel()
    z_values = info["heightscan"][..., 2].ravel()

    mean = jp.mean(z_values)
    std = jp.std(z_values)
    z_normal=z_values-jp.min(z_values)
    # z_normal = jp.where(std < 1e-8, jp.zeros_like(z_values), (z_values - mean) / std)   
    noisy_heightscan = (
      z_normal
      + (2 * jax.random.uniform(noise_rng, z_normal.shape) - 1)
      * self._config.noise_config.level
      * self._config.noise_config.scales.heightscan
    )  

    should_update = (info["step"] % self._config.history_update_steps) == 0

    qvel_history = jp.where(
        should_update,
        jp.roll(info["qvel_history"], 12).at[:12].set(data.qvel[6:]),
        info["qvel_history"],
    )

    qpos_error_history = jp.where(
        should_update,
        jp.roll(info["qpos_error_history"], 12).at[:12].set(data.qpos[7:] - info["motor_targets"]),
        info["qpos_error_history"],
    )

    info["qvel_history"] = qvel_history
    info["qpos_error_history"] = qpos_error_history

    state = jp.hstack([
        #noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        noisy_joint_angles - self._default_pose,  # 12
        noisy_joint_vel,  # 12
        phase,# 8
        # qvel_history,
        # qpos_error_history,
        noisy_heightscan, #N^2
        info["gait_freq"],# 1
        info["last_act"],  # 12
        info["command"],  # 3
    ])

    accelerometer = self.get_accelerometer(data)
    angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()

    privileged_state = jp.hstack([
        state,
        linvel,
        accelerometer,  # 3
        angvel,  # 3
        data.actuator_force,  # 12
        info["last_contact"],  # 4
        feet_vel,  # 4*3
        info["feet_air_time"],  # 4
        data.xfrc_applied[self._torso_body_id, :3],  # 3
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    return {
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data)
        ),
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "orientation": self._cost_orientation(self.get_upvector(data)),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        "termination": self._cost_termination(done),
        "pose": self._reward_pose(data.qpos[7:]),
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_clearance": self._cost_feet_clearance(data,info["phase"],info["H_max"]),
     
        "feet_phase":self._reward_feet_phase(
            data, info["phase"],info["H_max"]+self._config.reward_config.swing_height,self._config.reward_config.base_feet_distance
        ),
        "body_height":self._reward_body_height(
            data
        ),
        "feet_swing":self._reward_swing(
            data, info["phase"], self._config.reward_config.swing_height,info["H_max"]
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "contact":self._reward_contact(info["phase"],contact),
        "center":self._reward_center(data,self.init_feet_pos),
        "feet_height":self._cost_feet_height(info["swing_peak"], first_contact, info)
    }
  def _cost_feet_clearance(self, data: mjx.Data,phase: jax.Array,H_max) -> jax.Array:
    """Reward feet clearance when feet are in swing phase based on phi and H_max."""
    
    foot_pos_global= data.site_xpos[self._feet_site_id]
    foot_z_global=foot_pos_global[...,-1]
    x = phase / (2 * jp.pi)                        
    swing_mask = x >= 0.5
    clearance_mask = foot_z_global > H_max      

    reward = jp.where(swing_mask & clearance_mask, 1.0, 0.0)

    return jp.sum(reward)
  
