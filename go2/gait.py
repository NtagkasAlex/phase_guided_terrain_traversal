from typing import Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import matplotlib.pyplot as plt
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
if __name__=="__main__":
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