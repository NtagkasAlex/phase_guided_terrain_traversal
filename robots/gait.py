from typing import Union

import jax
import jax.numpy as jp
import numpy as np
from jax import lax

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
    foot_position = np.array(foot_position_value, dtype=np.float64)

    base_tf_offset_hip_joint = np.array([0.1934, 0.0465, 0.0], dtype=np.float64)

    if foot_num > 1:
        base_tf_offset_hip_joint[0] *= -1
    if foot_num % 2 == 1:
        base_tf_offset_hip_joint[1] *= -1

    delta = foot_position - base_tf_offset_hip_joint
    foot_distance = np.linalg.norm(delta)

    try:
        E = np.sqrt(max(foot_distance**2 - HIP_LENGTH**2, 1e-8))

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

        if np.isnan(J) or np.isnan(A) or np.isnan(S):
            return 0.0, 0.0, 0.0

        return np.array([J, A, S])

    except Exception:
        return 0.0, 0.0, 0.0

def joint_trajectory_np(
    phi: Union[jax.Array, float], swing_height: Union[jax.Array, float] = 0.08, swing_min: Union[jax.Array, float] = None
) -> jax.Array:
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
