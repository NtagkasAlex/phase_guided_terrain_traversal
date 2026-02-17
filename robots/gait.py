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


def get_robot_joints(foot_position_value, foot_num):
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


# # Link parameters per leg, ordered [FR, FL, RR, RL]
# # FR=RF in URDF, FL=LF, RR=RH, RL=LH
# ANYMAL_BASE_TO_HIP = jp.array([
#     [ 0.2999, -0.104,  0.0],   # FR
#     [ 0.2999,  0.104,  0.0],   # FL
#     [-0.2999, -0.104,  0.0],   # RR
#     [-0.2999,  0.104,  0.0],   # RL
# ])
# ANYMAL_HIP_TO_THIGH = jp.array([
#     [ 0.0599, -0.08381, 0.0],  # FR
#     [ 0.0599,  0.08381, 0.0],  # FL
#     [-0.0599, -0.08381, 0.0],  # RR
#     [-0.0599,  0.08381, 0.0],  # RL
# ])
# ANYMAL_THIGH_TO_SHANK = jp.array([
#     [0.0, -0.1003, -0.285],    # FR
#     [0.0,  0.1003, -0.285],    # FL
#     [0.0, -0.1003, -0.285],    # RR
#     [0.0,  0.1003, -0.285],    # RL
# ])
# ANYMAL_SHANK_TO_FOOT = jp.array([
#     [ 0.08795, -0.01305, -0.33797],  # FR
#     [ 0.08795,  0.01305, -0.33797],  # FL
#     [-0.08795, -0.01305, -0.33797],  # RR
#     [-0.08795,  0.01305, -0.33797],  # RL
# ])
# # Nominal foot x,y positions at zero config (base_to_hip + hip_to_thigh + thigh_to_shank + shank_to_foot)
# ANYMAL_NOMINAL_FOOT_POS = ANYMAL_BASE_TO_HIP + ANYMAL_HIP_TO_THIGH + ANYMAL_THIGH_TO_SHANK + ANYMAL_SHANK_TO_FOOT
# # X configuration: front knees bend backwards, rear knees bend forwards
# ANYMAL_KNEE_BEND_BWD = jp.array([True, True, False, False])


# def _anymal_ik(pos_base_to_foot, base_to_hip, hip_to_thigh, thigh_to_shank, shank_to_foot, knee_bend_bwd):
#     pos_hip_to_foot = pos_base_to_foot - base_to_hip
#     y_at_zero = (hip_to_thigh + thigh_to_shank + shank_to_foot)[1]

#     # qHAA
#     pos_yz_sq = pos_hip_to_foot[1]**2 + pos_hip_to_foot[2]**2
#     r_delta = jp.maximum(pos_yz_sq - y_at_zero**2, 0.0)
#     r = jp.sqrt(r_delta)
#     delta = jp.arctan2(pos_hip_to_foot[1], -pos_hip_to_foot[2])
#     beta = jp.arctan2(r, y_at_zero)
#     q_haa = beta + delta - jp.pi / 2

#     # Rotation matrix (HAA rotation about x-axis)
#     cos_haa = jp.cos(q_haa)
#     sin_haa = jp.sin(q_haa)
#     R_hip = jp.array([
#         [1.0, 0.0, 0.0],
#         [0.0, cos_haa, -sin_haa],
#         [0.0, sin_haa, cos_haa],
#     ])

#     pos_base_to_thigh = base_to_hip + R_hip @ hip_to_thigh
#     pos_thigh_to_foot = pos_base_to_foot - pos_base_to_thigh
#     d_kfe = jp.dot(pos_thigh_to_foot, pos_thigh_to_foot)

#     H = hip_to_thigh
#     T = thigh_to_shank
#     S = shank_to_foot

#     # KFE trigonometric equation: a*cos(x) + b*sin(x) = c
#     a_kfe = -S[0]*T[2]*2.0 + S[2]*T[0]*2.0
#     b_kfe =  S[0]*T[0]*2.0 + S[2]*T[2]*2.0
#     c_kfe = (S[1]*T[1]*2.0
#              + S[0]**2 + S[1]**2 + S[2]**2
#              + T[0]**2 + T[1]**2 + T[2]**2)
#     c_val = d_kfe - c_kfe
#     disc = jp.maximum(a_kfe**2 + b_kfe**2 - c_val**2, 0.0)
#     first_term = jp.arctan2(a_kfe, b_kfe)
#     second_term = jp.arctan2(jp.sqrt(disc), c_val)
#     q_kfe = jp.where(knee_bend_bwd, first_term - second_term, first_term + second_term)

#     # HFE trigonometric equation
#     d_hfe = jp.dot(pos_hip_to_foot, pos_hip_to_foot)
#     cos_kfe = jp.cos(q_kfe)
#     sin_kfe = jp.sin(q_kfe)

#     a_hfe = (H[0]*T[2]*2.0 - H[2]*T[0]*2.0
#              - H[0]*S[0]*sin_kfe*2.0 + H[0]*S[2]*cos_kfe*2.0
#              - H[2]*S[0]*cos_kfe*2.0 - H[2]*S[2]*sin_kfe*2.0)
#     b_hfe = (H[0]*T[0]*2.0 + H[2]*T[2]*2.0
#              + H[0]*S[0]*cos_kfe*2.0 + H[0]*S[2]*sin_kfe*2.0
#              - H[2]*S[0]*sin_kfe*2.0 + H[2]*S[2]*cos_kfe*2.0)
#     c_hfe = (S[1]*T[1]*2.0
#              + H[0]**2 + H[1]**2 + H[2]**2
#              + S[0]**2 + S[1]**2 + S[2]**2
#              + T[0]**2 + T[1]**2 + T[2]**2
#              + H[1]*S[1]*2.0 + H[1]*T[1]*2.0
#              + S[0]*T[0]*cos_kfe*2.0 - S[0]*T[2]*sin_kfe*2.0
#              + S[2]*T[0]*sin_kfe*2.0 + S[2]*T[2]*cos_kfe*2.0)
#     c_val_hfe = d_hfe - c_hfe
#     disc_hfe = jp.maximum(a_hfe**2 + b_hfe**2 - c_val_hfe**2, 0.0)
#     first_term_hfe = jp.arctan2(a_hfe, b_hfe)
#     second_term_hfe = jp.arctan2(jp.sqrt(disc_hfe), c_val_hfe)
#     q_hfe = jp.where(knee_bend_bwd, first_term_hfe + second_term_hfe, first_term_hfe - second_term_hfe)

#     joint_sum = q_haa + q_hfe + q_kfe
#     result = jp.where(jp.isnan(joint_sum), jp.zeros(3), jp.array([q_haa, q_hfe, q_kfe]))
#     return result


# @jax.jit
# def joint_trajectory_anymal(
#     phi: Union[jax.Array, float], swing_height: Union[jax.Array, float] = -0.4, swing_min: Union[jax.Array, float] = None
# ) -> jax.Array:
#     z = get_z(phi, swing_height=swing_height, swing_min=swing_min)
#     # Use nominal x,y from ANYmal kinematic chain; z from swing trajectory
#     p_fr = jp.array([ANYMAL_NOMINAL_FOOT_POS[0, 0], ANYMAL_NOMINAL_FOOT_POS[0, 1], z[0]])
#     p_fl = jp.array([ANYMAL_NOMINAL_FOOT_POS[1, 0], ANYMAL_NOMINAL_FOOT_POS[1, 1], z[1]])
#     p_rr = jp.array([ANYMAL_NOMINAL_FOOT_POS[2, 0], ANYMAL_NOMINAL_FOOT_POS[2, 1], z[2]])
#     p_rl = jp.array([ANYMAL_NOMINAL_FOOT_POS[3, 0], ANYMAL_NOMINAL_FOOT_POS[3, 1], z[3]])

#     return jp.array([
#         _anymal_ik(p_fr, ANYMAL_BASE_TO_HIP[0], ANYMAL_HIP_TO_THIGH[0], ANYMAL_THIGH_TO_SHANK[0], ANYMAL_SHANK_TO_FOOT[0], True),
#         _anymal_ik(p_fl, ANYMAL_BASE_TO_HIP[1], ANYMAL_HIP_TO_THIGH[1], ANYMAL_THIGH_TO_SHANK[1], ANYMAL_SHANK_TO_FOOT[1], True),
#         _anymal_ik(p_rr, ANYMAL_BASE_TO_HIP[2], ANYMAL_HIP_TO_THIGH[2], ANYMAL_THIGH_TO_SHANK[2], ANYMAL_SHANK_TO_FOOT[2], False),
#         _anymal_ik(p_rl, ANYMAL_BASE_TO_HIP[3], ANYMAL_HIP_TO_THIGH[3], ANYMAL_THIGH_TO_SHANK[3], ANYMAL_SHANK_TO_FOOT[3], False),
#     ]).reshape(-1)

# --- Kinematic Constants (LF, RF, LH, RH) ---
BASE_TO_HIP = jp.array([
    [0.2999, 0.104, 0.0], [0.2999, -0.104, 0.0],
    [-0.2999, 0.104, 0.0], [-0.2999, -0.104, 0.0]
])

HIP_TO_THIGH = jp.array([
    [0.0599, 0.08381, 0.0], [0.0599, -0.08381, 0.0],
    [-0.0599, 0.08381, 0.0], [-0.0599, -0.08381, 0.0]
])

THIGH_TO_SHANK = jp.array([
    [0.0, 0.1003, -0.285], [0.0, -0.1003, -0.285],
    [0.0, 0.1003, -0.285], [0.0, -0.1003, -0.285]
])

SHANK_TO_FOOT = jp.array([
    [0.08795, 0.01305, -0.33797], [0.08795, -0.01305, -0.33797],
    [-0.08795, 0.01305, -0.33797], [-0.08795, -0.01305, -0.33797]
])

# LF/RF: True (Knee back), LH/RH: False (Knee forward)
KNEE_BEND_BACKWARDS = jp.array([True, True, False, False])

def solve_linear_trig_jax(a, b, c):
    """Solves a*cos(x) + b*sin(x) = c"""
    delta = a**2 + b**2 - c**2
    delta = jp.maximum(0.0, delta)
    first_term = jp.atan2(a, b)
    second_term = jp.atan2(jp.sqrt(delta), c)
    return (first_term - second_term), (first_term + second_term)

def get_anymal_joints(foot_pos_value, foot_num):
    p_base_to_foot = jp.array(foot_pos_value)
    
    # Limb-specific selection
    b_to_h = BASE_TO_HIP[foot_num]
    h_to_t = HIP_TO_THIGH[foot_num]
    t_to_s = THIGH_TO_SHANK[foot_num]
    s_to_f = SHANK_TO_FOOT[foot_num]
    knee_back = KNEE_BEND_BACKWARDS[foot_num]

    # 1. HAA Calculation
    p_hip_to_foot = p_base_to_foot - b_to_h
    y_zero = h_to_t[1] + t_to_s[1] + s_to_f[1]
    
    r_yz_sq = p_hip_to_foot[1]**2 + p_hip_to_foot[2]**2
    r_delta = jp.maximum(0.0, r_yz_sq - y_zero**2)
    r = jp.sqrt(r_delta)
    
    delta_angle = jp.atan2(p_hip_to_foot[1], -p_hip_to_foot[2])
    beta = jp.atan2(r, y_zero)
    q_haa = beta + delta_angle - jp.pi/2

    # 2. KFE Calculation
    c_haa, s_haa = jp.cos(q_haa), jp.sin(q_haa)
    # Rotate h_to_t by q_haa to find thigh position in base frame
    p_base_to_thigh = b_to_h + jp.array([
        h_to_t[0],
        c_haa * h_to_t[1] - s_haa * h_to_t[2],
        s_haa * h_to_t[1] + c_haa * h_to_t[2]
    ])
    p_thigh_to_foot = p_base_to_foot - p_base_to_thigh
    d_kfe = jp.sum(p_thigh_to_foot**2)

    a_kfe = s_to_f[0] * t_to_s[2] * -2.0 + s_to_f[2] * t_to_s[0] * 2.0
    b_kfe = s_to_f[0] * t_to_s[0] * 2.0 + s_to_f[2] * t_to_s[2] * 2.0
    c_kfe = (s_to_f[1] * t_to_s[1] * 2.0 + jp.sum(s_to_f**2) + jp.sum(t_to_s**2))
    
    sol1_k, sol2_k = solve_linear_trig_jax(a_kfe, b_kfe, d_kfe - c_kfe)
    q_kfe = lax.select(knee_back, sol1_k, sol2_k)

    # 3. HFE Calculation
    d_hfe = jp.sum(p_hip_to_foot**2)
    t2, t3 = jp.cos(q_kfe), jp.sin(q_kfe)
    
    a_hfe = (h_to_t[0]*t_to_s[2]*2.0 - h_to_t[2]*t_to_s[0]*2.0 - 
             h_to_t[0]*s_to_f[0]*t3*2.0 + h_to_t[0]*s_to_f[2]*t2*2.0 - 
             h_to_t[2]*s_to_f[0]*t2*2.0 - h_to_t[2]*s_to_f[2]*t3*2.0)
    
    b_hfe = (h_to_t[0]*t_to_s[0]*2.0 + h_to_t[2]*t_to_s[2]*2.0 + 
             h_to_t[0]*s_to_f[0]*t2*2.0 + h_to_t[0]*s_to_f[2]*t3*2.0 - 
             h_to_t[2]*s_to_f[0]*t3*2.0 + h_to_t[2]*s_to_f[2]*t2*2.0)
    
    c_hfe = (s_to_f[1]*t_to_s[1]*2.0 + jp.sum(h_to_t**2) + jp.sum(s_to_f**2) + 
             jp.sum(t_to_s**2) + h_to_t[1]*s_to_f[1]*2.0 + h_to_t[1]*t_to_s[1]*2.0 + 
             s_to_f[0]*t_to_s[0]*t2*2.0 - s_to_f[0]*t_to_s[2]*t3*2.0 + 
             s_to_f[2]*t_to_s[0]*t3*2.0 + s_to_f[2]*t_to_s[2]*t2*2.0)

    sol1_h, sol2_h = solve_linear_trig_jax(a_hfe, b_hfe, d_hfe - c_hfe)
    q_hfe = lax.select(knee_back, sol2_h, sol1_h)

    result = jp.array([q_haa, q_hfe, q_kfe])
    return lax.select(jp.any(jp.isnan(result)), jp.zeros(3), result)

# --- Trajectory Function ---

@jax.jit
def joint_trajectory_anymal(
    phi: Union[jax.Array, float], 
    swing_height: Union[jax.Array, float] = 0.08, 
    swing_min: Union[jax.Array, float] = None
) -> jax.Array:
    # Use swing_min in the get_z call
    z = get_z(phi, swing_height=swing_height, swing_min=swing_min)
    
    # Nominal horizontal offsets for ANYmal C stance
    x_off = 0.40
    y_off = 0.30
    
    # Mapping to correct foot_num indices: 0:LF, 1:RF, 2:LH, 3:RH
    p_lf = jp.array([ x_off,  y_off, z[0]]) # LF
    p_rf = jp.array([ x_off, -y_off, z[1]]) # RF
    p_lh = jp.array([-x_off,  y_off, z[2]]) # LH
    p_rh = jp.array([-x_off, -y_off, z[3]]) # RH

    return jp.array([
        get_anymal_joints(p_lf, 0),
        get_anymal_joints(p_rf, 1),
        get_anymal_joints(p_lh, 2),
        get_anymal_joints(p_rh, 3)
    ]).reshape(-1)

if __name__ == "__main__":
    import mujoco
    import mujoco.viewer
    import time
    import argparse

    parser = argparse.ArgumentParser(description="Visualize gait oscillation")
    parser.add_argument("--robot", type=str, default="anymal", choices=["anymal", "go2"])
    parser.add_argument("--freq", type=float, default=1., help="Gait frequency in Hz")
    parser.add_argument("--swing-height", type=float, default=None, help="Swing peak z (negative, in base frame)")
    parser.add_argument("--stance-z", type=float, default=None, help="Stance z (negative, in base frame)")
    args = parser.parse_args()

    if args.robot == "anymal":
        xml_path = "anymal/xmls/scene_mjx_feetonly.xml"
        asset_dir = "anymal/xmls/"
        traj_fn = joint_trajectory_anymal
        swing_height = args.swing_height if args.swing_height is not None else -0.35
        stance_z = args.stance_z if args.stance_z is not None else -0.5
    else:
        xml_path = "go2/xmls/scene_mjx_feetonly.xml"
        asset_dir = "go2/xmls/"
        traj_fn = joint_trajectory
        swing_height = args.swing_height if args.swing_height is not None else -0.2
        stance_z = args.stance_z if args.stance_z is not None else -0.3

    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_xml = os.path.join(root, xml_path)
    full_asset_dir = os.path.join(root, asset_dir)

    model = mujoco.MjModel.from_xml_path(full_xml)
    data = mujoco.MjData(model)
    
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    home_qpos_base = data.qpos[:7].copy()

    phase = np.array(PHASES, dtype=np.float64)
    dt = model.opt.timestep
    phase_dt = 2 * np.pi * dt * args.freq

    print(f"Robot: {args.robot}")
    print(f"Freq: {args.freq} Hz, swing_height: {swing_height}, stance_z: {stance_z}")
    print(f"Phase dt per step: {phase_dt:.5f} rad")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            phase = (phase + phase_dt) % (2 * np.pi)
            q_target = traj_fn(phase, swing_height=swing_height, swing_min=stance_z)
            data.ctrl[:] = q_target
            mujoco.mj_step(model, data)
            data.qpos[:7] = home_qpos_base
            data.qvel[:6] = 0.0
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(dt)
