from ml_collections import config_dict

import go2.go2_constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=1000,
      vel_percentage=0.65,
      Kp=40.0,
      Kd=0.5,
      action_repeat=1,
      action_scale=0.5,
      history_len=2,
      history_update_steps=5,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
              heightscan=0.01
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking.
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.5,
              # Base reward.
              lin_vel_z=-1.0,
              ang_vel_xy=-0.05,
              orientation=-0.2,
              # Other.
              dof_pos_limits=-1.0,
              pose=-1.0,
              # Other.
              termination=-1.0,
              stand_still=-0.0,
              # Regularization.
              torques=-0.0002,
              action_rate=-0.01,
              energy=-0.0005,
              # Feet.
              feet_clearance=-0.,
              feet_height=-0.,
              feet_slip=-0.,
              feet_air_time=0.,
              feet_phase=0.5, #0.5,
              feet_swing=0.,
              body_height=-0.,
              contact=2., #2.,
              center=-0.
          ),
          tracking_sigma=0.2,
          swing_height=-0.2,
          base_feet_distance=-0.3,
        #   swing_height=-0.25,
        #   base_feet_distance=-0.35,
          phase_sigma=0.05,
      ),

      command_config=config_dict.create(
          # Uniform distribution for command amplitude.
          u_max=[1.5, 0.8, 1.2],
          u_min=[-1.5,- 0.8, -1.2],
          # a=[0.7, 0., 0.],
          # Probability of not zeroing out new command.
          b=[0.9, 0.25, 0.5],
      ),
      gait_freq=[2,6],
      heighmap_size=(consts.num_heightscans,consts.num_widthscans),
      
  )


def baseline_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      vel_percentage=0.65,
      episode_length=1000,
      Kp=40.0,
      Kd=0.5,
      action_repeat=1,
      action_scale=0.5,
      history_len=2,
      history_update_steps=5,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
              heightscan=0.01
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking.
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.5,
              # Base reward.
              lin_vel_z=-2.0,
              ang_vel_xy=-0.05,
              orientation=-0.2,
              # Other.
              dof_pos_limits=-1.0,
              pose=-0.2,
              # Other.
              termination=-1.0,
              stand_still=-0.5,
              # Regularization.
              torques=-0.0002,
              action_rate=-0.005,
              energy=-0.0005,
              # Feet.
              feet_clearance=-1.0,#0.25,
              feet_height=-0.,
              feet_slip=-0.1,
              feet_air_time=0.1,
              feet_phase=0.0, #0.5,
              feet_swing=0.,
              body_height=-0.,
              contact=0., #2.,
              center=-0.
          ),
          tracking_sigma=0.25,
          swing_height=-0.2,
          base_feet_distance=-0.3,
          phase_sigma=0.05,
      ),
      command_config=config_dict.create(
          # Uniform distribution for command amplitude.
          u_max=[1.5, 0.8, 1.2],
          u_min=[-1.5,- 0.8, -1.2],
          # a=[0.7, 0., 0.],
          # Probability of not zeroing out new command.
          b=[0.9, 0.25, 0.5],
      ),
      gait_freq=[2,6],
      heighmap_size=(consts.num_heightscans,consts.num_widthscans),
      
  )
