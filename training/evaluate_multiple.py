#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from types import SimpleNamespace
import copy
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import your training function
from train import *
eval_metrics_all=[]
def progress_eval(num_steps, metrics,env_cfg):
    # clear_output(wait=True)

    times.append(datetime.now())
    x_data.append(num_steps)
    
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])

    vel_tracking=metrics["eval/episode_reward/tracking_lin_vel"]#/(env_cfg.reward_config.scales.tracking_lin_vel* env_cfg.episode_length)
    ang_tracking=metrics["eval/episode_reward/tracking_ang_vel"]#/(env_cfg.reward_config.scales.tracking_ang_vel* env_cfg.episode_length)
    xy_ang_vel=metrics["eval/episode_reward/ang_vel_xy"]/env_cfg.reward_config.scales.ang_vel_xy
  
    term_vals=metrics["eval/final_reward/termination"]
    count = int(jnp.sum(jnp.abs(term_vals) < 0.5))
    survival_rate = count / 1000

    energy = metrics["eval/episode_reward/torques"]/ env_cfg.reward_config.scales.torques 
    

    eval_metrics_all.append({
        "tracking_lin_vel": vel_tracking,
        "tracking_ang_vel": ang_tracking,
        "xy_ang_vel": xy_ang_vel,
        "energy": energy,
        "survival_rate": survival_rate,
    })

    return True



def sweep(method: str, ckpt_folder: str):
    results = []
    eval_metrics_all.clear()  # Clear before each sweep

    for i in range(1,2):
        terrain = f"terrains/level{i:02d}.npy"
        args = SimpleNamespace(
            method=method,
            checkpoint_folder=ckpt_folder,
            task_name="stairs",
            terrain_file=terrain,
            num_envs=4096,
            batch_size=256,
            discount=0.97,
            learning_rate=3e-4,
            num_minibatches=32,
            num_timesteps=1,
            num_evals=1,
            num_eval_envs=1000,
            index=0,
        )
        print(f"→ [{method}] running on {terrain} …", end="", flush=True)
        run_training(args,progress_eval)
    return copy.deepcopy(eval_metrics_all)
def main():
    # wild_results = sweep("wild", "checks_stairs/checkpoint_41")
    pgtt_results = sweep("pgtt", "checks_stairs/checkpoint_30")

    np.save("plots/wild_results.npy", np.array(wild_results, dtype=object))
    np.save("plots/pgtt_results.npy", np.array(pgtt_results, dtype=object))

    print("\nSaved:")
    print(" • wild_results.npy")
    print(" • pgtt_results.npy")

if __name__ == "__main__":
    main()