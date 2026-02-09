import mujoco
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mujoco.viewer import launch_passive
import time
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

import robots
import robots.base as robot_base
import robots.gait as gait
from functools import partial as _partial
import policy_net as pn
import torch
from cpu_heightmap.heightmap import create_sensor_matrix as _create_sensor_matrix_raw
import copy
import argparse

_parser = argparse.ArgumentParser(description="Deploy heightmap")
_parser.add_argument('--robot', type=str, default='go2', help='Robot name: go2 or anymal')
_args, _ = _parser.parse_known_args()
robot_cfg = robots.get_robot_config(_args.robot)
consts = robot_cfg
create_sensor_matrix = _partial(
    _create_sensor_matrix_raw,
    dist_x=consts.dist_x,
    dist_y=consts.dist_y,
    num_heightscans=consts.num_heightscans,
    num_widthscans=consts.num_widthscans,
)

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist

    class CmdVelSubscriber(Node):
        def __init__(self):
            super().__init__('cmd_vel_listener')
            self.subscription = self.create_subscription(
                Twist,
                '/cmd_vel',
                self.cmd_vel_callback,
                10)
            self.latest_twist = Twist()  

        def cmd_vel_callback(self, msg):
            self.latest_twist = msg  

        def get_velocity(self):
            
            return np.array([self.latest_twist.linear.x,self.latest_twist.linear.y,self.latest_twist.angular.z])
except ImportError:
    print("Running without ros...")


perturbation_mode = "random"
perturbation_mode = "fixed"


perturbation_type="sinusoidal"
perturbation_type = "constant"

fixed_pert_velocity = 1.
fixed_pert_duration = 1.0
fixed_pert_wait = 1.0
fixed_pert_angle = -0.4

_deploy_defaults = robot_cfg.DEPLOY_DEFAULTS
pert_enable = _deploy_defaults["pert_enable"]
command=np.array(_deploy_defaults["command"])
stairs_enabled=_deploy_defaults["stairs_enabled"]
gait_freq=_deploy_defaults["gait_freq"]
ros_enabled=False
render=True
feet_scans=False

mode="pgtt"
filename=_deploy_defaults["policy_path"]

total_time=50


class Controller:
    def __init__(self,policy_path,config_dict,default_pose,n_substeps,dt):
        self.config=config_dict
        self._default_angles=default_pose
        self._last_action=np.zeros(12)
        self.dt=dt
        self.policy_network=pn.policy_net(policy_file=policy_path)
        if ros_enabled:
            rclpy.init(args=None)
            self.node = CmdVelSubscriber()
        self._counter=0
        self._n_substeps=n_substeps
        self.phase=gait.PHASES
        self.heightscan=np.zeros((consts.num_heightscans,consts.num_heightscans,3))
        self.command=command
        self.motor_targets=np.zeros(12)
    def get_obs(self,model, data) -> np.ndarray:
        linvel = data.sensor("local_linvel").data
        gyro = data.sensor("gyro").data
        imu_xmat = data.site_xmat[model.site("imu").id].reshape((3, 3))
        gravity = imu_xmat.T @ np.array([0, 0, -1])
        joint_angles = data.qpos[7:] - self._default_angles
        joint_velocities = data.qvel[6:]
        self.phase+=2*np.pi*gait_freq*self.dt
        
        self.phase=np.fmod(self.phase,2*np.pi)
        
        cos = np.cos(self.phase)
        sin = np.sin(self.phase)
        phase = np.concatenate([cos, sin])

        yaw = np.arctan2(imu_xmat[1, 0], imu_xmat[0, 0])

        heightscan=create_sensor_matrix(model,data,data.qpos[:3],yaw)
        self.heightscan=heightscan
        z_values = heightscan[..., 2].ravel()
        mean = np.mean(z_values)
        std = np.std(z_values)

        if std<1e-8:
            z_normal=z_values*0
        else:
            z_normal = (z_values - mean) / std
        z_normal=z_values-np.min(z_values)

        if ros_enabled and self._counter % self._n_substeps == 0:
            rclpy.spin_once(self.node)
            self.command= self.node.get_velocity()*np.array(_deploy_defaults["ros_velocity_scale"])
       
        
        if mode=="baseline":
            obs = np.hstack([
            gyro,
            gravity,
            joint_angles,
            joint_velocities,
            z_normal,
            self._last_action,
            self.command, 
            ])
        else:
            obs = np.hstack([
                gyro,
                gravity,
                joint_angles,
                joint_velocities,
                phase,
                z_normal,
                gait_freq,
                self._last_action,
                self.command, 
            ])
        return obs.astype(np.float64)
    
    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
      
        if self._counter%self._n_substeps==0:
            obs = self.get_obs(model, data)
            obs_tensor=torch.tensor(np.asarray(obs).copy(), dtype=torch.float32).reshape((1,-1))
            prediction=self.policy_network(obs_tensor)#.to(device)
            
            self._last_action=np.copy(prediction.cpu().detach().numpy().reshape(-1))
            if mode=="wild":
                oscilator_angles = gait.joint_trajectory_np(self.phase,self.config.reward_config.swing_height,self.config.reward_config.base_feet_distance)
                data.ctrl= oscilator_angles + self.config.action_scale * self._last_action
            else:
                data.ctrl=self._default_angles + self.config.action_scale * self._last_action
            self.motor_targets=np.copy(self._default_angles + self.config.action_scale * self._last_action)
        self._counter += 1

def draw_joystick_command(
    scn,
   heightscan,
   xyz,
   theta,
   cmd,
   start_idx=0,
   color=None

):
    N= heightscan.shape[0]
    M=N
    dot_size = 0.015


    indices = np.arange(N * M)
    positions = heightscan.reshape(-1, 3)
    final_idx=0
    for idx, pos in zip(indices, positions):
        final_idx=start_idx+idx
        scn.geoms[start_idx+idx].category = mujoco.mjtCatBit.mjCAT_DECOR
        mujoco.mjv_initGeom(
            geom=scn.geoms[start_idx+idx],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[dot_size] * 3,
            pos=pos,
            mat=np.eye(3).ravel(),
            rgba=[0, 1, 1, 0.5] if color is None else color,
        )
    rgba = [0.2, 0.2, 0.6, 0.8]
    radius=0.02
    vx, vy, vtheta = cmd

    angle = theta + vtheta
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )

    arrow_from = xyz
    rotated_velocity = rotation_matrix @ np.array([vx, vy])
    to = np.asarray([rotated_velocity[0], rotated_velocity[1], 0])
    to = to / (np.linalg.norm(to) + 1e-6)
    arrow_to = arrow_from + to * 1

    mujoco.mjv_initGeom(
        geom=scn.geoms[scn.ngeom - 1 ],
        type=mujoco.mjtGeom.mjGEOM_ARROW.value,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.zeros(9),
        rgba=np.asarray(rgba).astype(np.float32),
    )
    mujoco.mjv_connector(
        geom=scn.geoms[scn.ngeom - 1],
        type=mujoco.mjtGeom.mjGEOM_ARROW.value,
        width=radius,
        from_=arrow_from,
        to=arrow_to,
    )      

config_dict=robot_cfg.default_config()
if stairs_enabled:
    model = mujoco.MjModel.from_xml_path(
        consts.TEST_STAIRS_XML.as_posix(),
        assets=robot_base.get_assets(consts),
    )
else:
    model = mujoco.MjModel.from_xml_path(
        consts.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
        assets=robot_base.get_assets(consts),
    )

model.dof_damping[6:] = config_dict.Kd
model.actuator_gainprm[:, 0] = config_dict.Kp   
model.actuator_biasprm[:, 1] = -config_dict.Kp

data = mujoco.MjData(model)
data.qpos= model.keyframe("home").qpos
default_pose = model.keyframe("home").qpos[7:]


mujoco.mj_step(model, data)

heightscan=create_sensor_matrix(model,data,data.qpos[:3],0)
data.qpos[2]+=np.max(heightscan[...,2])


ctrl_dt = config_dict.ctrl_dt
sim_dt = config_dict.sim_dt
n_substeps = int(round(ctrl_dt / sim_dt))
model.opt.timestep = sim_dt


#Pertubation related
torso_body_id = model.body(consts.ROOT_BODY).id
torso_mass = model.body_subtreemass[torso_body_id]


pert_velocity_range = [4., 5.0]
pert_duration_range = [0.6, 1.8]
pert_wait_range = [0.2, 2.0]

pert_steps_since_last = 0
pert_steps_until_next = int(np.random.uniform(*pert_wait_range) / sim_dt)
pert_duration_steps = 0
pert_mag = 0.0
pert_dir = np.zeros(3)
pert_active = False

def maybe_apply_perturbation():
    global pert_steps_since_last, pert_steps_until_next
    global pert_duration_steps, pert_mag, pert_dir, pert_active

    if perturbation_mode == "fixed":
        velocity_range = [fixed_pert_velocity, fixed_pert_velocity]
        duration_range = [fixed_pert_duration, fixed_pert_duration]
        wait_range = [fixed_pert_wait, fixed_pert_wait]

    else: 
        velocity_range = pert_velocity_range
        duration_range = pert_duration_range
        wait_range = pert_wait_range

    if pert_active:
        if perturbation_type == "sinusoidal":
            t = pert_steps_since_last * sim_dt
            u_t = 0.5 * np.sin(np.pi * t / (pert_duration_steps * sim_dt))
        else:  
            u_t = 1.0  
        force = u_t * torso_mass * pert_mag / (pert_duration_steps * sim_dt)
        data.xfrc_applied[torso_body_id, :3] = force * pert_dir

        pert_steps_since_last += 1
        if pert_steps_since_last >= pert_duration_steps:
            pert_active = False
            pert_steps_since_last = 0
            pert_steps_until_next = int(np.random.uniform(*wait_range) / sim_dt)
            data.xfrc_applied[torso_body_id, :3] = 0.0
    else:
        pert_steps_until_next -= 1
        if pert_steps_until_next <= 0:
            pert_active = True
            pert_steps_since_last = 0
            pert_duration_steps = int(np.random.uniform(*duration_range) / sim_dt)
            pert_mag = np.random.uniform(*velocity_range)
            if perturbation_mode == "fixed":
                angle = fixed_pert_angle
            else:
                angle = np.random.uniform(0, 2 * np.pi)
            pert_dir = np.array([np.cos(angle), np.sin(angle), 0.0])


control=Controller(policy_path=filename,config_dict=config_dict,default_pose=default_pose,n_substeps=n_substeps,dt=ctrl_dt)

paused = False
select=-1
def key_callback(keycode):
    if chr(keycode) == ' ':
        global paused
        paused = not paused
    if chr(keycode) == 'O':
        print(select)


if render:
    viewer = launch_passive(model, data,key_callback=key_callback,show_left_ui=False,show_right_ui=False)
    viewer.user_scn.ngeom+=consts.num_heightscans**2 +1 
    if pert_enable:
        viewer.user_scn.ngeom+=1
    viewer.user_scn.flags[mujoco.mjtRndFlag. mjRND_SHADOW] = 0
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = 1
    viewer.sync()
start_time = time.time()


from robots.gait import joint_gait
trajectory=[]
while data.time <total_time:#config_dict.episode_length*sim_dt:
    if pert_enable:
        maybe_apply_perturbation()

    control.get_control(model,data)
    command=control.command

    heightscan=control.heightscan
    n = (heightscan.shape[0] - 1) // 2  # This gives us the value of 'n' based on the shape of heightscan

    imu_xmat = data.site_xmat[model.site("imu").id].reshape((3, 3))
    yaw = np.arctan2(imu_xmat[1, 0], imu_xmat[0, 0])
    xyz=data.qpos[:3]+np.array([0,0,0.2])
    if not paused:
        # print(data.time)
        mujoco.mj_step(model, data)
        # heightscan[...,2]=z_normal.reshape((9,9))
        # print(.shape)
        if render:

            draw_joystick_command(viewer.user_scn,control.heightscan,xyz,yaw,command,start_idx=0)   
            if pert_active:
                arrow_from = data.xpos[torso_body_id]
                scale_factor = 1. # meters per m/s of velocity_kick
                arrow_to = arrow_from + pert_dir * (pert_mag * scale_factor)
                mujoco.mjv_initGeom(
                    geom=viewer.user_scn.geoms[viewer.user_scn.ngeom - 2],
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.zeros(9),
                    rgba=np.array([1, 0, 0, 0.8], dtype=np.float32),
                )
                mujoco.mjv_connector(
                    geom=viewer.user_scn.geoms[viewer.user_scn.ngeom - 2],
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    width=0.02,
                    from_=arrow_from,
                    to=arrow_to,
                )    
            else:
                mujoco.mjv_initGeom(
                    geom=viewer.user_scn.geoms[viewer.user_scn.ngeom - 2],
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.zeros(9),
                    rgba=np.array([1, 0, 0, 0.0], dtype=np.float32),
                )
            
            viewer.sync()
    time.sleep(0.002)
