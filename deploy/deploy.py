import mujoco

from mujoco.viewer import launch_passive
import time
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

import go2.base as go2_base
import go2.joystick as joystick
import go2.go2_constants as consts
import policy_net as pn
import torch
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import go2.gait as gait
class CmdVelSubscriber(Node):
    def __init__(self):
        super().__init__('cmd_vel_listener')
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)
        self.latest_twist = Twist()  # Store latest velocity message

    def cmd_vel_callback(self, msg):
        self.latest_twist = msg  # Update latest message

    def get_velocity(self):
        
        return np.array([self.latest_twist.linear.x,self.latest_twist.linear.y,self.latest_twist.angular.z])


command=np.array([0.5,0.,0.])
gait_freq=5
class Controller:
    def __init__(self,policy_path,config_dict,default_pose,n_substeps,dt):
        self.config=config_dict
        self._default_angles=default_pose
        self._last_action=default_pose
        self.dt=dt
        self.policy_network=pn.policy_net(policy_file=policy_path)
        self._counter=0
        self._n_substeps=n_substeps
        rclpy.init(args=None)
        self.node = CmdVelSubscriber()
        self.phase=gait.PHASES

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
        #Comment these out for passive control
        # rclpy.spin_once(self.node)
        # command=            self.node.get_velocity()
        obs = np.hstack([
            # linvel,
            gyro,
            gravity,
            joint_angles,
            joint_velocities,
            # phase,
            self._last_action,
            command,
        ])
        # print(obs.shape)
        return obs.astype(np.float32)
    
    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._counter += 1
        if self._counter % self._n_substeps == 0:
            obs = self.get_obs(model, data)
            obs_tensor=torch.tensor(np.asarray(obs).copy(), dtype=torch.float32).reshape((1,-1))
            prediction=self.policy_network(obs_tensor)
            
            self._last_action=prediction.cpu().detach().numpy().reshape(-1)
            # print(action)
            data.ctrl[:]=self._default_angles + self.config.action_scale * self._last_action
        

config_dict=joystick.default_config()

model = mujoco.MjModel.from_xml_path(
      consts.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
      assets=go2_base.get_assets(),
  )

data = mujoco.MjData(model)
data.qpos= model.keyframe("home").qpos
default_pose = model.keyframe("home").qpos[7:]
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)

ctrl_dt = config_dict.ctrl_dt
sim_dt = config_dict.sim_dt
n_substeps = int(round(ctrl_dt / sim_dt))
model.opt.timestep = sim_dt

control=Controller(policy_path="policy25",config_dict=config_dict,default_pose=default_pose,n_substeps=n_substeps,dt=sim_dt)


render=True

if render:
    viewer = launch_passive(model, data)
    
    viewer.sync()
start_time = time.time()


while data.time < 30.:
    
    control.get_control(model,data)
    data.ctrl=default_pose
    mujoco.mj_step(model, data)
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2
        dist = contact.dist
        if dist>0:
            print(f"Contact {i}: geom1 = {geom1}, geom2 = {geom2}, dist = {dist:.6f}")
    if render:
        viewer.sync()
if render:
    viewer.close()
