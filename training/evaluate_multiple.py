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



def sweep(method: str, ckpt_folder: str, ckpt_id: int, height_cm: int):
    eval_metrics_all.clear() # Clear before each sweep


    terrain = f"terrains/level{height_cm:02d}.npy"
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
    eval_flag=True,
    )
    print(f"→ [{method}] ckpt {ckpt_id} running on {terrain} …", end="", flush=True)
    run_training(args, progress_eval)
    return copy.deepcopy(eval_metrics_all)
def sweep_discrete(method: str, ckpt_folder: str):
    results = []
    eval_metrics_all.clear()  # Clear before each sweep

    terrain = f"terrains/discrete.npy"
    args = SimpleNamespace(
        method=method,
        checkpoint_folder=ckpt_folder,
        task_name="stairs",
        terrain_file=terrain,
        num_envs=4096,
        batch_size=4096,
        discount=0.97,
        learning_rate=3e-4,
        num_minibatches=8,
        num_timesteps=1,
        num_evals=1,
        num_eval_envs=1000,
        index=0,
        eval_flag=True, 
    )
    print(f"→ [{method}] running on {terrain} …", end="", flush=True)
    run_training(args,progress_eval)
    return copy.deepcopy(eval_metrics_all)
def main():

    checkpoints = {
    "pgtt": [ "checks_stairs/checkpoint_93", "checks_stairs/checkpoint_113", "checks_stairs/checkpoint_133","checks_stairs/checkpoint_153"],
    "baseline": [ "checks_stairs/checkpoint_97", "checks_stairs/checkpoint_117", "checks_stairs/checkpoint_137","checks_stairs/checkpoint_157"],
    "wild": [ "checks_stairs/checkpoint_101", "checks_stairs/checkpoint_121", "checks_stairs/checkpoint_141","checks_stairs/checkpoint_161"],
    }
    # checkpoints = {
    # "pgtt": [ "checks_stairs/checkpoint_93", "checks_stairs/checkpoint_113", "checks_stairs/checkpoint_133"],
    # "baseline": [ "checks_stairs/checkpoint_97", "checks_stairs/checkpoint_117", "checks_stairs/checkpoint_137"],
    # "wild": [ "checks_stairs/checkpoint_101", "checks_stairs/checkpoint_121", "checks_stairs/checkpoint_141"],
    # }

    Path("plots").mkdir(exist_ok=True)


    for method, ckpt_list in checkpoints.items():
        for ckpt_idx, ckpt_path in enumerate(ckpt_list, start=1):
            for height in range(1, 10): # 1 cm to 10 cm
                print(f"→ [{method}] ckpt {ckpt_idx} on height {height}cm …", end="", flush=True)
                # print()
                results = sweep(method, ckpt_path, ckpt_idx, height)
                filename = f"plots/{method}_dist_ckpt{ckpt_idx}_h{height:02d}.npy"
                np.save(filename, np.array(results, dtype=object))
                print(f" saved {filename}")


    print("\nAll evaluations complete. Results saved in plots/.")
    # pgtt_results = sweep("pgtt", "checks_stairs/checkpoint_73")
    # baseline_results=sweep("baseline", "checks_stairs/checkpoint_77")
    # wild_results = sweep("wild", "checks_stairs/checkpoint_81")

    

    # np.save("plots/wild_results_1.npy", np.array(wild_results, dtype=object))
    # np.save("plots/baseline_results_1.npy", np.array(baseline_results, dtype=object))
    # np.save("plots/pgtt_results_1.npy", np.array(pgtt_results, dtype=object))

    # pgtt_results = sweep_discrete("pgtt", "checks_stairs/checkpoint_72")
    # baseline_results=sweep_discrete("baseline", "checks_stairs/checkpoint_76")
    # wild_results = sweep_discrete("wild", "checks_stairs/checkpoint_80")

    
    # np.save("plots/wild_results_discrete.npy", np.array(wild_results, dtype=object))
    # np.save("plots/baseline_results_discrete.npy", np.array(baseline_results, dtype=object))
    # np.save("plots/pgtt_results_discrete.npy", np.array(pgtt_results, dtype=object))


if __name__ == "__main__":
    main()