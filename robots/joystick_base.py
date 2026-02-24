from typing import Any, Dict, Optional, Union
import functools

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

# from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
import robots.base as robot_base
import robots.gait as gait
from robots.heightmap import create_sensor_matrix as _create_sensor_matrix


class Joystick_Base(robot_base.RobotEnv):
  """Track a joystick command."""

  def __init__(
      self,
      consts,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = None,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    if config is None:
      config = consts.default_config()
    super().__init__(
        consts=consts,
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    # Bind robot-specific heightmap parameters
    self._heightmap_fn = functools.partial(
        _create_sensor_matrix,
        dist_x=consts.dist_x,
        dist_y=consts.dist_y,
        num_heightscans=consts.num_heightscans,
        num_widthscans=consts.num_widthscans,
    )

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)

    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.1, maxval=0.1)
    )

    data = mjx_env.make_data(
        self.mjx_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[7:],
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    #Use this if some envs start from stairs or level 1
    heightscan=self._heightmap_fn(self.mjx_model,data,data.qpos[:3],0)
    qpos = qpos.at[2].set(qpos[2] +jp.max(heightscan[...,2]))
    data = data.replace(qpos=qpos)
    data = mjx.forward(self.mjx_model, data)

    rng, key1, key2, key3 = jax.random.split(rng, 4)
    time_until_next_pert = jax.random.uniform(
        key1,
        minval=self._config.pert_config.kick_wait_times[0],
        maxval=self._config.pert_config.kick_wait_times[1],
    )
    steps_until_next_pert = jp.round(time_until_next_pert / self.dt).astype(
        jp.int32
    )
    pert_duration_seconds = jax.random.uniform(
        key2,
        minval=self._config.pert_config.kick_durations[0],
        maxval=self._config.pert_config.kick_durations[1],
    )
    pert_duration_steps = jp.round(pert_duration_seconds / self.dt).astype(
        jp.int32
    )
    pert_mag = jax.random.uniform(
        key3,
        minval=self._config.pert_config.velocity_kick[0],
        maxval=self._config.pert_config.velocity_kick[1],
    )

    rng, key1, key2 = jax.random.split(rng, 3)
    time_until_next_cmd = jax.random.exponential(key1) * 5.0

    steps_until_next_cmd = jp.round(time_until_next_cmd / self.dt).astype(
        jp.int32
    )

    cmd = jax.random.uniform(
        key2, shape=(3,), minval=self._cmd_u_min, maxval=self._cmd_u_max
    )
    rng, key = jax.random.split(rng)

    gait_freq = jax.random.uniform(
        key,  minval=self._config.gait_freq[0], maxval=self._config.gait_freq[1]
    )
    qpos_error_history = jp.zeros(self._config.history_len * 12)
    qvel_history = jp.zeros(self._config.history_len * 12)
    heightscan=self._heightmap_fn(self.mjx_model,data,data.qpos[:3],0)
    info = {
        "rng": rng,
        "command": cmd,
        "step":0,
        "steps_until_next_cmd": steps_until_next_cmd,
        "phase":gait.PHASES,
        "phase_dt":2*jp.pi*self.dt*gait_freq,
        "gait_freq":gait_freq,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(4),
        "last_contact": jp.zeros(4, dtype=bool),
        "swing_peak": jp.zeros(4),
        "H_max":0.1*jp.ones(4),
        "heightscan":heightscan,
        "H_min":0.*jp.ones(4),
        "motor_targets":0.*jp.ones(12),
        "qpos_error_history": qpos_error_history,
        "qvel_history": qvel_history,
        "steps_until_next_pert": steps_until_next_pert,
        "pert_duration_seconds": pert_duration_seconds,
        "pert_duration": pert_duration_steps,
        "steps_since_last_pert": 0,
        "pert_steps": 0,
        "pert_dir": jp.zeros(3),
        "pert_mag": pert_mag,
        "perturbation_type": self._config.pert_config.type,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    self._weights=np.array([1.0,0.1,0.1]*4)
    self.init_feet_pos.at[:].set(self.get_feet_pos(data))
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)
    motor_targets = self._default_pose + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    # contact = self.compute_contact(data,self._feet_geom_id, self._floor_geom_id)
    contact = jp.array([
        data.sensordata[self.mjx_model.sensor_adr[sensorid]] > 0
        for sensorid in self._feet_floor_found_sensor
    ])

    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = self.get_feet_pos(data)
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    heightscan=self._heightmap_fn(self.mjx_model,data,data.qpos[:3],self.get_yaw(data))
    n = (heightscan.shape[0] - 1) // 2
    top_right = heightscan[:n, n+1:,2]
    top_left = heightscan[:n, :n,2]
    back_right = heightscan[n+1:, n+1:,2]
    back_left = heightscan[n+1:, :n,2]

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
    state.info["H_max"]=H_max_value-H_min_value
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

    cos = jp.cos(info["phase"])
    sin = jp.sin(info["phase"])
    phase = jp.concatenate([cos, sin])

    z_values = info["heightscan"][..., 2].ravel()

    mean = jp.mean(z_values)
    std = jp.std(z_values)
    z_normal=z_values-jp.min(z_values)
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
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        noisy_joint_angles - self._default_pose,  # 12
        noisy_joint_vel,  # 12
        phase,# 8
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
        "feet_clearance": self._cost_feet_clearance(data,info["H_max"]),

        "feet_phase":self._reward_feet_phase(
            data, info["phase"],info["H_max"]+self._config.reward_config.swing_height,self._config.reward_config.base_feet_distance
        ),
        "body_height":self._reward_body_height(
            data,info
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
  def _reward_center(self,data,init):
    feet_pos=self.get_feet_pos(data)
    weight=jp.array([1.,1.,0.])
    return jp.sum(jp.square(feet_pos-init)*weight)

  def _reward_body_height(self,data,info):
      body_height=data.qpos[2]
      heightscan=info["heightscan"]
      center_heightscan=heightscan[heightscan.shape[0]//2,heightscan.shape[1]//2,2]
      return jp.sum((body_height-center_heightscan-self._config.reward_config.base_feet_distance)**2)
  def _reward_contact(self,phase,contact):
      x = phase / (2 * jp.pi)
      stance_mask = x < gait.p_stance
      swing_mask = x >= gait.p_stance


      reward_mask = stance_mask & contact
      penalty_mask = swing_mask & contact

      reward_terms = reward_mask.astype(jp.float32)
      penalty_terms = penalty_mask.astype(jp.float32)

      reward = - jp.sum(penalty_terms)
      return reward

  def _reward_swing(
    self, data: mjx.Data, phase: jax.Array, p_des: jax.Array,H_max
    ) -> jax.Array:
        foot_pos_local=self.get_feet_pos(data)
        foot_z_local=foot_pos_local[...,-1]
        x = phase / (2 * jp.pi)

        swing_mask = x >= gait.p_stance  # swing phase
        reward_mask=jp.where(swing_mask,1.0,0.0)
        reward=jp.sum(jp.square(foot_z_local - p_des) * reward_mask)
        return reward

  def _reward_feet_phase(
      self, data: mjx.Data, phase: jax.Array, swing_height: jax.Array,swing_min:jax.Array
  ) -> jax.Array:

    foot_pos_local=self.get_feet_pos(data)
    foot_z_local=foot_pos_local[...,-1]
    rz = gait.get_z(phase, swing_height=swing_height,swing_min=swing_min)
    error = jp.sum(jp.square(foot_z_local - rz))
    return jp.exp(-error /  self._config.reward_config.phase_sigma)

  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes).
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of angular velocity commands (yaw).
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  # Base-related rewards.

  def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
    # Penalize z axis base linear velocity.
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    # Penalize xy axes base angular velocity.
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    # Penalize non flat base orientation.
    return jp.sum(jp.square(torso_zaxis[:2]))

  # Energy related rewards.

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    # Penalize torques.
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    # Penalize energy consumption.
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))

  # Other rewards.

  def _reward_pose(self, qpos: jax.Array) -> jax.Array:
    # Stay close to the default pose.
    return jp.sum(jp.square(qpos - self._default_pose) * self._weights)

  def _cost_stand_still(
      self,
      commands: jax.Array,
      qpos: jax.Array,
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.01)

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    # Penalize early termination.
    return done

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    # Penalize joints if they cross soft limits.
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  # Feet related rewards.

  def _cost_feet_slip(
      self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(info["command"])
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
    return jp.sum(vel_xy_norm_sq * contact) * (cmd_norm > 0.01)

  def _cost_feet_clearance(self, data: mjx.Data,H_max) -> jax.Array:
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos_local=self.get_feet_pos(data)
    foot_z_local=foot_pos_local[...,-1]
    delta = jp.abs(foot_z_local - (H_max+self._config.reward_config.swing_height))

    return jp.sum(delta * vel_norm)

  def _cost_feet_height(
      self,
      swing_peak: jax.Array,
      first_contact: jax.Array,
      info: dict[str, Any],
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(info["command"])
    error = swing_peak / self._config.reward_config.swing_height - 1.0
    return jp.sum(jp.square(error) * first_contact) * (cmd_norm > 0.01)

  def _reward_feet_air_time(
      self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
  ) -> jax.Array:
    # Reward air time.
    cmd_norm = jp.linalg.norm(commands)
    rew_air_time = jp.sum((air_time - 0.5) * first_contact)
    rew_air_time *= cmd_norm > 0.01  # No reward for zero commands.
    return rew_air_time

  def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
    def gen_dir(rng: jax.Array) -> jax.Array:
        angle = jax.random.uniform(rng, minval=0.0, maxval=jp.pi * 2)
        return jp.array([jp.cos(angle), jp.sin(angle), 0.0])

    def apply_pert(state: mjx_env.State) -> mjx_env.State:
        t = state.info["pert_steps"] * self.dt

        # constant (0) vs sinusoidal (1)
        u_t = jp.where(
            state.info["perturbation_type"] == 0,
            1.0,  # constant scaling
            0.5 * jp.sin(jp.pi * t / state.info["pert_duration_seconds"]),  # sinusoidal
        )

        # Force: kg * m/s * 1/s = N
        force = (
            u_t
            * self._torso_mass
            * state.info["pert_mag"]
            / state.info["pert_duration_seconds"]
        )

        xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
        xfrc_applied = xfrc_applied.at[self._torso_body_id, :3].set(
            force * state.info["pert_dir"]
        )

        data = state.data.replace(xfrc_applied=xfrc_applied)
        state = state.replace(data=data)

        # reset counter if perturbation ends
        state.info["steps_since_last_pert"] = jp.where(
            state.info["pert_steps"] >= state.info["pert_duration"],
            0,
            state.info["steps_since_last_pert"],
        )
        state.info["pert_steps"] += 1
        return state

    def wait(state: mjx_env.State) -> mjx_env.State:
        state.info["rng"], rng = jax.random.split(state.info["rng"])
        state.info["steps_since_last_pert"] += 1

        xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
        data = state.data.replace(xfrc_applied=xfrc_applied)

        # Reset counters + pick new direction if starting new perturbation
        state.info["pert_steps"] = jp.where(
            state.info["steps_since_last_pert"] >= state.info["steps_until_next_pert"],
            0,
            state.info["pert_steps"],
        )
        state.info["pert_dir"] = jp.where(
            state.info["steps_since_last_pert"] >= state.info["steps_until_next_pert"],
            gen_dir(rng),
            state.info["pert_dir"],
        )
        return state.replace(data=data)

    return jax.lax.cond(
        state.info["steps_since_last_pert"] >= state.info["steps_until_next_pert"],
        apply_pert,
        wait,
        state,
    )


  def sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
    rng, y_rng, w_rng, z_rng = jax.random.split(rng, 4)
    y_k = jax.random.uniform(
        y_rng, shape=(3,), minval=self._cmd_u_min, maxval=self._cmd_u_max
    )
    z_k = jax.random.bernoulli(z_rng, self._cmd_b, shape=(3,))
    w_k = jax.random.bernoulli(w_rng, 0.5, shape=(3,))
    x_kp1 = x_k - w_k * (x_k - y_k * z_k)
    return x_kp1
