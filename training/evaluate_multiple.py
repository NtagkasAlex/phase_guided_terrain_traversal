#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from types import SimpleNamespace
import copy
import sys, os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import your training function
from train import *

_parser = argparse.ArgumentParser(description="Evaluate multiple checkpoints")
_parser.add_argument('--robot', type=str, default='go2', help='Robot name: go2 or anymal')
_parser.add_argument('--eval_type', type=str, default='both', help='Evaluation type: stairs, discrete, or both')
_parser.add_argument('--run', type=int, default=0, help='Which training run to evaluate')
_parser.add_argument('--height_min', type=int, default=1, help='Min stair height in cm (for stairs eval)')
_parser.add_argument('--height_max', type=int, default=10, help='Max stair height in cm (for stairs eval)')
_eval_args, _ = _parser.parse_known_args()
ROBOT = _eval_args.robot

eval_metrics_all=[]
def progress_eval(num_steps, metrics,env_cfg):
    times.append(datetime.now())
    x_data.append(num_steps)

    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])

    vel_tracking=metrics["eval/episode_reward/tracking_lin_vel"]
    ang_tracking=metrics["eval/episode_reward/tracking_ang_vel"]
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
    eval_metrics_all.clear()

    terrain = f"terrains/level{height_cm:02d}.npy"
    args = SimpleNamespace(
        robot=ROBOT,
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
        index="eval",
        eval_flag=True,
    )
    print(f"  [{method}] ckpt {ckpt_id} running on {terrain} …", end="", flush=True)
    run_training(args, progress_eval)
    return copy.deepcopy(eval_metrics_all)

def sweep_discrete(method: str, ckpt_folder: str):
    eval_metrics_all.clear()

    terrain = f"terrains/discrete.npy"
    args = SimpleNamespace(
        robot=ROBOT,
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
        index="eval",
        eval_flag=True,
    )
    print(f"  [{method}] running on {terrain} …", end="", flush=True)
    run_training(args, progress_eval)
    return copy.deepcopy(eval_metrics_all)

def main():
    METHODS = ["pgtt", "baseline", "wild"]
    LEVELS = ["level03", "level07", "level10", "level13"]
    RUN = _eval_args.run
    eval_type = _eval_args.eval_type
    height_min = _eval_args.height_min
    height_max = _eval_args.height_max

    checkpoints = {
        method: [
            f"checks_stairs/checkpoint_{method}_{level}_run{RUN}"
            for level in LEVELS
        ]
        for method in METHODS
    }

    Path("plots").mkdir(exist_ok=True)

    if eval_type in ("discrete", "both"):
        print(f"=== Discrete terrain evaluation (robot={ROBOT}, run={RUN}) ===")
        for method, ckpt_list in checkpoints.items():
            for ckpt_idx, ckpt_path in enumerate(ckpt_list, start=1):
                print(f"  [{method}] ckpt {ckpt_idx} on discrete terrain …", end="", flush=True)
                results = sweep_discrete(method, ckpt_path)
                filename = f"plots/{method}_discrete_ckpt{ckpt_idx}.npy"
                np.save(filename, np.array(results, dtype=object))
                print(f" saved {filename}")
        print("\nDiscrete evaluations complete.")

    if eval_type in ("stairs", "both"):
        print(f"\n=== Stair height evaluation (robot={ROBOT}, run={RUN}, heights={height_min}-{height_max}cm) ===")
        for method, ckpt_list in checkpoints.items():
            for ckpt_idx, ckpt_path in enumerate(ckpt_list, start=1):
                for height in range(height_min, height_max + 1):
                    results = sweep(method, ckpt_path, ckpt_idx, height)
                    filename = f"plots/{method}_dist_ckpt{ckpt_idx}_h{height:02d}.npy"
                    np.save(filename, np.array(results, dtype=object))
                    print(f" saved {filename}")
        print("\nStair evaluations complete.")

    print("\nAll evaluations complete. Results saved in plots/.")


if __name__ == "__main__":
    main()
