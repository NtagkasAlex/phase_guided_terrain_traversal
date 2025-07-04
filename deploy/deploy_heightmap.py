import mujoco

from mujoco.viewer import launch_passive
import time
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

import go2.base as go2_base
import go2.joystick_pgtt as joystick
import go2.go2_constants as consts
import policy_net as pn
import torch
from cpu_heightmap.heightmap import create_sensor_matrix,create_feet_heightscan
import go2.gait as gait
import copy

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

command=np.array([0.4,0,0.])
stairs_enabled=True
gait_freq=2.0
ros_enabled=False
render=True
feet_scans=False
baseline=False
total_time=50
filename="./policy/perceptive/go2_no_cpg" # idk
filename="./policy/blind/go2_height0" #good for flat
# filename="./policy/perceptive/go2_after_fine_only_swing" #best so far
# filename="go2_new5"
# filename="go2_w_dr_50m"
filename="go2_w_dr_100m"

filename="i_believe"
# filename="test"

filename="policy_folder/policy3"
# filename="policy19"
filename="policy_folder/policy177"
# /
filename="policy_folder/policy183"
# filename="flat_0"


class Controller:
    def __init__(self,policy_path,config_dict,default_pose,n_substeps,dt):
        self.config=config_dict
        self._default_angles=default_pose
        self._last_action=np.zeros(12)
        self.dt=dt
        self.policy_network=pn.policy_net(policy_file=policy_path)
        # self.policy_network.to(device)
        if ros_enabled:
            rclpy.init(args=None)
            self.node = CmdVelSubscriber()
        self._counter=0
        self._n_substeps=n_substeps
        self.phase=gait.PHASES
        self.heightscan=np.zeros((consts.num_heightscans,consts.num_heightscans,3))
        self.command=command
        self.qvel_history = np.zeros(12 * 2)  # 10-step history of 12 joint vels
        self.qpos_error_history = np.zeros(12 * 2)  # 10-step history of joint pos errors
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
        self.heightscan=heightscan#+(data.qpos[2]-0.28)
        z_values = heightscan[..., 2].ravel()
        # z_values=np.zeros(13*9)
        # z_values=z_values[:81]
        mean = np.mean(z_values)
        std = np.std(z_values)

        if std<1e-8:
            z_normal=z_values*0
        else:
            z_normal = (z_values - mean) / std
        z_normal=z_values-np.min(z_values)
        # z_normal=z_values
        # print(heightscan[...,2])
        # heightscan=-0.05*np.ones((consts.num_heightscans,consts.num_heightscans,3))
        # heightscan=hmap.create_sensor_matrix(data.qpos[:3],0.)
        #Comment these out for passive control
        # print(self._counter)
        if ros_enabled and self._counter % self._n_substeps == 0:
            rclpy.spin_once(self.node)
            self.command= self.node.get_velocity()*np.array([0.5,0.5,1.0])
        # --- History tracking ---
       
        qpos_error = data.qpos[7:]  - (self._default_angles + self.config.action_scale * self._last_action)

        # --- History tracking ---
        self.qvel_history = np.roll(self.qvel_history, 12)
        self.qvel_history[:12] = joint_velocities

        self.qpos_error_history = np.roll(self.qpos_error_history, 12)
        self.qpos_error_history[:12] = qpos_error 
        
        if baseline:
            obs = np.hstack([
            # linvel,
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
                # linvel,
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
       
        # print(self.qvel_history)
        #     state = np.hstack([
        #     linvel,  # 3
        #     gyro,  # 3
        #     gravity,  # 3
        #     joint_angles - self._default_pose,  # 12
        #     joint_vel,  # 12
        #     phase,# 8
        #     info["heightscan"][...,2].ravel(),# N^2
        #     info["gait_freq"],
        #     info["last_act"],  # 12
        #     info["command"],  # 3
        # ])
        return obs.astype(np.float64)
    
    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        # if self._counter==0:
        #     time.sleep(3.)
        if self._counter%self._n_substeps==0:
            obs = self.get_obs(model, data)
            obs_tensor=torch.tensor(np.asarray(obs).copy(), dtype=torch.float32).reshape((1,-1))
            # obs_tensor.to(device)
            prediction=self.policy_network(obs_tensor)#.to(device)
            
            self._last_action=np.copy(prediction.cpu().detach().numpy().reshape(-1))
            # print(action)
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
        geom=scn.geoms[scn.ngeom - 1],
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

config_dict=joystick.default_config()
if stairs_enabled:
    model = mujoco.MjModel.from_xml_path(
        consts.TEST_STAIRS_XML.as_posix(),
        assets=go2_base.get_assets(),
    )
else:
    model = mujoco.MjModel.from_xml_path(
        consts.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
        assets=go2_base.get_assets(),
    )

# model = mujoco.MjModel.from_xml_path(
#       consts.HUGE_STAIRS_XML.as_posix(),
#       assets=go2_base.get_assets(),
# )
# model = mujoco.MjModel.from_xml_path("./go2/xmls/scene_mjx.xml")
# config_dict.Kp=50
model.dof_damping[6:] = config_dict.Kd
model.actuator_gainprm[:, 0] = config_dict.Kp   
model.actuator_biasprm[:, 1] = -config_dict.Kp

data = mujoco.MjData(model)
data.qpos= model.keyframe("home").qpos
default_pose = model.keyframe("home").qpos[7:]
# data.qpos[0]+=2.5
# data.qpos[1]+=.45


mujoco.mj_step(model, data)

heightscan=create_sensor_matrix(model,data,data.qpos[:3],0)
data.qpos[2]+=np.max(heightscan[...,2])
print(np.max(heightscan))


# data = mujoco.MjData(model)
# mujoco.mj_resetDataKeyframe(model, data, 0)

ctrl_dt = config_dict.ctrl_dt
print(ctrl_dt)
sim_dt = config_dict.sim_dt
# ctrl_dt = sim_dt
n_substeps = int(round(ctrl_dt / sim_dt))
model.opt.timestep = sim_dt

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
    if feet_scans:
        viewer.user_scn.ngeom+=4*9
    viewer.user_scn.flags[mujoco.mjtRndFlag. mjRND_SHADOW] = 0
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = 1
    viewer.sync()
start_time = time.time()

print(model.ngeom)
FEET_SITES = [
    "FR_foot",
    "FL_foot",
    "RR_foot",
    "RL_foot",
]
_feet_site_id = np.array(
        [model.site(name).id for name in FEET_SITES]
    )

print(default_pose)
print(model.ngeom)
# print(imu_id)
_save=[]
# data.qpos[2]=0.4
# 
_data=[]
from go2.gait import joint_gait
trajectory=[]
while data.time <total_time:#config_dict.episode_length*sim_dt:
    foot_pos = data.site_xpos[_feet_site_id]  
    foot_z = foot_pos[..., -1]        

    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'FR_foot')
    start_idx = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    accel = data.sensordata[start_idx:start_idx+dim]
    # print("Accel:", accel)
    # print(data.qpos[7:10])
    # select=viewer._pert.select
    # print(data.site_xpos[0])  
    # _data.append(foot_z[1])
    _data.append(foot_z[0])
    trajectory.append(copy.copy(data))
    control.get_control(model,data)
    command=control.command
    # _save.append(np.copy(data.ctrl[1:3]))
    # print(data.ctrl[1:3])
    # data.ctrl=
    # print(joint_gait(control.phase))
    # print(control.phase)
    # data.ctrl=default_pose
    # print(data.ncon)
    # print(joint_gait(control.phase)[1:3]+default_pose[1:3])
    # data.ctrl=joint_gait(control.phase)+default_pose
# 
    # print(data.ctrl[1:3],
    # data.ctrl[4:6],
    # data.ctrl[7:9],
    # data.ctrl[10:12])

    heightscan=control.heightscan
    n = (heightscan.shape[0] - 1) // 2  # This gives us the value of 'n' based on the shape of heightscan
    # Extracting the four regions of the heightscan matrix
    # top_right = heightscan[:n, n+1:,2]          # Top-right corner
    # top_left = heightscan[:n, :n,2]             # Top-left corner
    # back_right = heightscan[n+1:, n+1:,2]       # Back-right corner
    # back_left = heightscan[n+1:, :n,2]         # Back-left corner
    # back_left = heightscan[:n, n+1:,2]          # Top-right corner
    # back_right = heightscan[:n, :n,2]             # Top-left corner
    # top_left = heightscan[n+1:, n+1:,2]       # Back-right corner
    # top_right = heightscan[n+1:, :n,2]         # Back-left corner

        # Flatten the data
    
    # print(z_normal)
    
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
            # if feet_scans:
            #     draw_joystick_command(viewer.user_scn,create_feet_heightscan(model,data,foot_pos[0]),start_idx=consts.num_heightscans**2+0 ,color=[0.5,0.5,0,1])    
            #     draw_joystick_command(viewer.user_scn,create_feet_heightscan(model,data,foot_pos[1]),start_idx=consts.num_heightscans**2+9 ,color=[1,0,0,1])    
            #     draw_joystick_command(viewer.user_scn,create_feet_heightscan(model,data,foot_pos[2]),start_idx=consts.num_heightscans**2+18,color=[0,1,0,1])    
            #     draw_joystick_command(viewer.user_scn,create_feet_heightscan(model,data,foot_pos[3]),start_idx=consts.num_heightscans**2+27,color=[0.5,0,0.5,1])    
            #     pass

            viewer.sync()
    time.sleep(0.002)
plt.plot(_data)
plt.show()
# np.save("joint_traj",_save)
# if render:
    # viewer.close()





# image_list = []

# from matplotlib.animation import FuncAnimation

# with mujoco.Renderer(model) as renderer:
#     for _data in trajectory:
#         renderer.update_scene(_data)
#         img = renderer.render()
#         image_list.append(img)

# # Create a figure and axis
# fig, ax = plt.subplots()

# # This function will be called to update the frame during the animation
# def update(frame):
#     ax.clear()         # Clear previous frame
#     ax.imshow(image_list[frame])  # Display the current image
#     ax.axis('off')     # Hide axis for clean visualization

# # Create the animation
# ani = FuncAnimation(fig, update, frames=len(image_list), interval=1, repeat=False)

# # Display the animation
# plt.show()
