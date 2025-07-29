from typing import Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import matplotlib.pyplot as plt
import go2.base as go2_base
import go2.go2_constants as consts
from jax import lax
from mujoco.viewer import launch_passive
# from configs import *
# URDF GO2 real values
HIP_LENGTH = 0.0955
THIGH_LENGTH = 0.213
CALF_LENGTH = 0.2135

p_stance=0.5
PHASES=jp.array([0.,jp.pi,jp.pi,0.])

def cubic_hermite(t, p0, p1, m0, m1):
    t2 = t ** 2
    t3 = t ** 3
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2

    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
def spline_func(p0, p1, m0, m1, T=1.):
    return lambda t: cubic_hermite(t / T, p0, p1, T * m0, T * m1)

from typing import Union
import jax.numpy as jp
import jax

def get_z(
    phi: Union[jax.Array, float], swing_height: Union[jax.Array, float] = 0.08, swing_min: Union[jax.Array, float] = None
) -> jax.Array:
    h_max = swing_height
    if swing_min is None:
        stance = jp.zeros_like(phi)  
    else:
        stance=swing_min
    x=phi
    T_swing=2*jp.pi*(1-p_stance)/2
    T_peak = 2*jp.pi*(1 + p_stance) / 2  
    T_stance=2*jp.pi*p_stance
    swing_up = spline_func(stance, h_max, 0, 0,T_swing)
    swing_down = spline_func(h_max, stance, 0, 0,T_swing)

    
    return jp.where(
        x <= T_stance, stance, 
        jp.where(
            x <= T_peak, swing_up((x - T_stance)), 
            swing_down((x - T_peak))
        )
    )
def get_swing(
    phi: Union[jax.Array, float], swing_height: Union[jax.Array, float] = 0.08, swing_min: Union[jax.Array, float] = None
) -> jax.Array:
    # x = phi / (2 * jp.pi)  # Normalize phi to [0,1]
    h_max = swing_height
    mid_point = (1 + p_stance) / 2  
    if swing_min is None:
        stance = jp.zeros_like(phi)  
    else:
        stance=swing_min
    swing_up = spline_func(stance, h_max, 0, 0)
    swing_down = spline_func(h_max, stance, 0, 0)
   
    T_swing=(1-p_stance)/2
   
    return jp.where(
        x <= p_stance, stance, 
        jp.where(
            x <= mid_point, swing_up(1/T_swing*(x - p_stance)), 
            swing_down(1/T_swing*(x - mid_point))
        )
    )
def joint_gait(phi, scale=0.3, beta=0.5):
    
    f_T_swing = 1 / (2 * (1 - beta))
    _t = phi / (2 * (1 - beta))
    signal = scale * jp.sin(phi * f_T_swing)



    true_output = jp.stack([jp.zeros_like(signal), -0.2*jp.sin(phi * f_T_swing)+0.1, 0.4 * jp.sin(phi * f_T_swing)], axis=-1)
    

    false_output = jp.zeros_like(true_output)


    condition = (phi < 2 * (1 - beta) * jp.pi) & (phi > 0)

    result = jp.where(condition[..., None], true_output, false_output)

    return true_output.reshape(-1)

def get_robot_joints(foot_position_value, foot_num):
    # Convert foot position to a JAX array
    foot_position = jp.array(foot_position_value)

    # Compute base_tf_offset_hip_joint
    base_tf_offset_hip_joint = jp.array([0.1934, 0.0465, 0.0])
    base_tf_offset_hip_joint = base_tf_offset_hip_joint.at[0].set(
        lax.select(foot_num > 1, -base_tf_offset_hip_joint[0], base_tf_offset_hip_joint[0])
    )
    base_tf_offset_hip_joint = base_tf_offset_hip_joint.at[1].set(
        lax.select(foot_num % 2 == 1, -base_tf_offset_hip_joint[1], base_tf_offset_hip_joint[1])
    )

    # Distance calculation
    delta = foot_position - base_tf_offset_hip_joint
    foot_position_distance = jp.linalg.norm(delta)

    # Law of cosines for angle calculation
    E = jp.sqrt(jp.maximum(foot_position_distance ** 2 - HIP_LENGTH ** 2, 1e-8))  # prevent sqrt of negative

    y = jp.arccos(
        jp.clip((E ** 2 + THIGH_LENGTH ** 2 - CALF_LENGTH ** 2) / (2 * E * THIGH_LENGTH), -1.0, 1.0)
    )
    S = jp.arccos(
        jp.clip((CALF_LENGTH ** 2 + THIGH_LENGTH ** 2 - E ** 2) / (2 * CALF_LENGTH * THIGH_LENGTH), -1.0, 1.0)
    ) - jp.pi

    C = delta[0]
    R = delta[1]

    A = lax.select(
        foot_position[2] < 0,
        jp.arcsin(jp.clip(-C / E, -1.0, 1.0)) + y,
        -jp.pi + jp.arcsin(jp.clip(C / E, -1.0, 1.0)) + y
    )

    O = jp.sqrt(jp.maximum(foot_position_distance ** 2 - C ** 2, 1e-8))
    L = jp.arcsin(jp.clip(R / O, -1.0, 1.0))

    P = lax.select(foot_position[2] > 0, -1.0, 1.0)
    hip_angle = jp.arcsin(jp.clip(HIP_LENGTH / O, -1.0, 1.0))

    J = lax.select(
        foot_num % 2 == 0,
        P * (L - hip_angle),
        P * (L + hip_angle)
    )

    # Handle NaNs safely
    joint_sum = J + A + S
    safe_output = (0.0, 0.0, 0.0)
    result = lax.select(jp.isnan(joint_sum), jp.array(safe_output), jp.array([J, A, S]))

    return result
@jax.jit
def joint_trajectory(
    phi: Union[jax.Array, float], swing_height: Union[jax.Array, float] = 0.08, swing_min: Union[jax.Array, float] = None
) -> jax.Array:
    """
    Generate joint angles for a gait trajectory based on the phase angle phi.
    
    Args:
        phi: Phase angle in radians.
        swing_height: Height of the swing phase.
        swing_min: Minimum height during the stance phase.
    
    Returns:
        Joint angles for the gait trajectory.
    """
    z = get_z(phi, swing_height=swing_height, swing_min=swing_min)
    x_off=0.15
    y_off=0.15
    p1=jp.array([x_off,-y_off,z[0]])#FR
    p2=jp.array([x_off,y_off,z[1]])#FL
    p3=jp.array([x_off,-y_off,z[2]])#RR
    p4=jp.array([x_off,y_off,z[3]])#RL


    return jp.array([
        get_robot_joints(p1, 1),  # Front Right
        get_robot_joints(p2, 0),  # Front Left
        get_robot_joints(p3, 1),  # Rear Right
        get_robot_joints(p4, 0)   # Rear Left
    ]).reshape(-1)


def get_robot_joints_np(foot_position_value, foot_num):
    """
    Compute inverse kinematics for a single robotic leg.
    
    Args:
        foot_position_value: list or array of (x, y, z) position of the foot.
        foot_num: int, foot index (used to flip offsets depending on side).
    
    Returns:
        Tuple of joint angles (hip, thigh, knee) in radians.
    """
    foot_position = np.array(foot_position_value, dtype=np.float64)

    # Base transform offset from robot center to hip joint
    base_tf_offset_hip_joint = np.array([0.1934, 0.0465, 0.0], dtype=np.float64)

    # Flip directions based on foot number
    if foot_num > 1:
        base_tf_offset_hip_joint[0] *= -1
    if foot_num % 2 == 1:
        base_tf_offset_hip_joint[1] *= -1

    # Relative vector from hip joint to foot position
    delta = foot_position - base_tf_offset_hip_joint
    foot_distance = np.linalg.norm(delta)

    # Distance projected into the leg plane
    try:
        E = np.sqrt(max(foot_distance**2 - HIP_LENGTH**2, 1e-8))  # avoid sqrt(negative)

        # Law of cosines
        y = np.arccos(np.clip((E**2 + THIGH_LENGTH**2 - CALF_LENGTH**2) / (2 * E * THIGH_LENGTH), -1.0, 1.0))
        S = np.arccos(np.clip((CALF_LENGTH**2 + THIGH_LENGTH**2 - E**2) / (2 * CALF_LENGTH * THIGH_LENGTH), -1.0, 1.0)) - np.pi

        C = delta[0]
        R = delta[1]

        if foot_position[2] < 0:
            A = np.arcsin(np.clip(-C / E, -1.0, 1.0)) + y
        else:
            A = -np.pi + np.arcsin(np.clip(C / E, -1.0, 1.0)) + y

        O = np.sqrt(max(foot_distance**2 - C**2, 1e-8))
        L = np.arcsin(np.clip(R / O, -1.0, 1.0))

        P = -1 if foot_position[2] > 0 else 1
        hip_offset = np.arcsin(np.clip(HIP_LENGTH / O, -1.0, 1.0))

        if foot_num % 2 == 0:
            J = P * (L - hip_offset)
        else:
            J = P * (L + hip_offset)

        # NaN protection
        if np.isnan(J) or np.isnan(A) or np.isnan(S):
            return 0.0, 0.0, 0.0

        return np.array([J, A, S])

    except Exception:
        return 0.0, 0.0, 0.0
    
def joint_trajectory_np(
    phi: Union[jax.Array, float], swing_height: Union[jax.Array, float] = 0.08, swing_min: Union[jax.Array, float] = None
) -> jax.Array:
    """
    Generate joint angles for a gait trajectory based on the phase angle phi.
    
    Args:
        phi: Phase angle in radians.
        swing_height: Height of the swing phase.
        swing_min: Minimum height during the stance phase.
    
    Returns:
        Joint angles for the gait trajectory.
    """
    z = get_z(phi, swing_height=swing_height, swing_min=swing_min)
    x_off=0.15
    y_off=0.15
    p1=np.array([x_off,-y_off,z[0]])#FR
    p2=np.array([x_off,y_off,z[1]])#FL
    p3=np.array([x_off,-y_off,z[2]])#RR
    p4=np.array([x_off,y_off,z[3]])#RL


    return np.array([
        get_robot_joints_np(p1, 1),  # Front Right
        get_robot_joints_np(p2, 0),  # Front Left
        get_robot_joints_np(p3, 1),  # Rear Right
        get_robot_joints_np(p4, 0)   # Rear Left
    ]).reshape(-1)



if __name__=="__main__":
    model = mujoco.MjModel.from_xml_path(
        consts.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
        assets=go2_base.get_assets(),
    )
    # config_dict=default_config()
    model.dof_damping[6:] = 0.5
    model.actuator_gainprm[:, 0] = 40
    model.actuator_biasprm[:, 1] = -40

    data = mujoco.MjData(model)
    data.qpos= model.keyframe("home").qpos
    default_state=model.keyframe("home").qpos
    default_pose = model.keyframe("home").qpos[7:]
    # data.qpos[0]+=2.5

    # data.qpos[2]=0.4

    mujoco.mj_step(model, data)

    viewer = launch_passive(model, data)
    phi=np.array([0.0,np.pi,np.pi,0])
    dt=0.01
    f=2
    model.opt.timestep = dt

    while data.time< 10:
        phi+=dt*2*np.pi*f
        phi=np.fmod(phi,2*jp.pi)
    # #     # foot_num=0
    #     foot_num=1
    #     z=get_z(phi, swing_height=jp.array([-0.2]), swing_min=jp.array([-0.3]))
    #     # print(z)
    #     x_off=0.15
    #     y_off=0.15
    #     p1=np.array([x_off,-y_off,z[0]])#FR
    #     p2=np.array([x_off,y_off,z[1]])#FL
    #     p3=np.array([x_off,-y_off,z[2]])#RR
    #     p4=np.array([x_off,y_off,z[3]])#RL
        data.ctrl=joint_trajectory(phi, swing_height=jp.array([-0.2]), swing_min=jp.array([-0.3]))
        # data.ctrl=default_pose    
        # data.ctrl[:3]=get_robot_joints_np(p1, 1) 
        # data.ctrl[3:6]=get_robot_joints_np(p2,0) 
        # data.ctrl[6:9]=get_robot_joints_np(p3, 1) 
        # data.ctrl[9:12]=get_robot_joints_np(p4, 0) 
        # p=np.array([0.25,0.1,-0.25])

        # data.ctrl[3:6]=get_robot_joints_np(p, 0) 

        # data.qpos[2]+=.1

        mujoco.mj_step(model, data)
        # data.qpos[:7]=default_state[:7]
        # data.qpos[2]=0.4
# 
        viewer.sync()
        
    exit()
    
    exit()
    # x=np.linspace(0,2*jp.pi,500)
    x=np.load("pgtt_lift_times.npy")[300:600]
    x=np.load("baseline_lift_times.npy")[0:300]

    phases=jp.array([0,jp.pi,0,jp.pi])
    phases=jp.array([0])
    plt.figure(figsize=(8, 5))
    plt.ylim([-0.4,-0.1])
    plt.ylim([-0.01,0.12])

    # data=np.load("joint_traj.npy")
    # print(data.shape)
    data=np.load("pgtt_lift.npy")[300:600]
    data=np.load("baseline_lift.npy")[0:300]

    for phase in phases:

        y = [get_z(jp.fmod(2*jp.pi*2*_x + phase,2*jp.pi),swing_height=jp.array([0.1]),swing_min=jp.array([0.])) for _x in x] 
        # y = [joint_gait(jp.fmod(6*_x + phase,2*jp.pi))[1:3]+np.array([0.9,-1.8]) for _x in x] 
        plt.plot(x,data,label="Actual Leg Trajectory")
        # plt.plot(x, y, label="Desired Leg Trajectory")
    # plt.plot(x, data[:500, 0], label='Element 1')
    # plt.plot(x, data[:500, 1], label='Element 2')

   
    # plt.show()

    swing_start = 2*p_stance*jp.pi
    swing_end = 2* jp.pi
    swing_height = -0.32

    plt.annotate("", xy=(swing_end, swing_height), xytext=(swing_start, swing_height),
                arrowprops=dict(arrowstyle="<->", color="red", linewidth=1.5))

    
    plt.text((swing_start + swing_end) / 2, swing_height + 0.005, "Swing Phase", 
            color="red", ha="center", fontsize=12)
    
    swing_start = 0.
    swing_end = 2*p_stance*jp.pi
    swing_height = -0.32 

    plt.annotate("", xy=(swing_end, swing_height), xytext=(swing_start, swing_height),
                arrowprops=dict(arrowstyle="<->", color="red", linewidth=1.5))

    # Adding text label
    plt.text((swing_start + swing_end) / 2, swing_height + 0.005, "Stance Phase", 
            color="red", ha="center", fontsize=12)

    plt.xlabel("Time(s)")
    plt.ylabel("Swing Height")
    plt.legend()
    plt.grid()

    grad = np.gradient(data, x)

# make the plot
    # plt.figure(figsize=(8, 5))
    # plt.plot(x, data, label="Actual Leg Trajectory")
    plt.plot(x, grad, label="Gradient d(data)/d(x)", linestyle="--")
    plt.xlabel("x (time)")
    plt.ylabel("Value / Gradient")
    # plt.ylim(-0.5, 0.5)         # adjust as needed
    plt.legend()
    plt.grid()

    plt.title("Trajectory and Its Gradient")

    plt.show()