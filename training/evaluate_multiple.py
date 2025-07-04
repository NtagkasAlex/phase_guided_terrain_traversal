#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from types import SimpleNamespace

# import your training function
from evaluate import run_training

def sweep(method: str, ckpt_folder: str):
    results = []
    for i in range(9, 12):
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
            num_evals=31,
            index=0,
        )
        print(f"→ [{method}] running on {terrain} …", end="", flush=True)
        r = run_training(args)
        print(f" {r:.3f}")
        results.append(r)
    return results

def main():
    pgtt_results = sweep("pgtt",     "checks_stairs/checkpoint_122")
    base_results = sweep("baseline", "checks_stairs/checkpoint_125")

    np.save("plots/pgtt_results.npy",     np.array(pgtt_results))
    np.save("plots/baseline_results.npy", np.array(base_results))

    print("\nSaved:")
    print(" • pgtt_results.npy")
    print(" • baseline_results.npy")

if __name__ == "__main__":
    main()
