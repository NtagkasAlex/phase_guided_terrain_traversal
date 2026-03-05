import distutils.util
import os
import subprocess

if subprocess.run('nvidia-smi').returncode:
  raise RuntimeError(
      'Cannot communicate with GPU. '
      'Make sure you are using a GPU Colab runtime. '
      'Go to the Runtime menu and select Choose runtime type.'
  )

# Add an ICD config so that glvnd can pick up the Nvidia EGL driver.
# This is usually installed as part of an Nvidia driver package, but the Colab
# kernel doesn't install its driver via APT, and as a result the ICD is missing.
# (https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/icd_enumeration.md)
NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
  with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
    f.write("""{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
""")

try:
  print('Checking that the installation succeeded:')
  import mujoco

  mujoco.MjModel.from_xml_string('<mujoco/>')
except Exception as e:
  raise e from RuntimeError(
      'Something went wrong during installation. Check the shell output above '
      'for more information.\n'
      'If using a hosted Colab runtime, make sure you enable GPU acceleration '
      'by going to the Runtime menu and selecting "Choose runtime type".'
  )

print('Installation successful.')

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

# @title Import packages for plotting and creating graphics
import json
import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np

# @title Import MuJoCo, MJX, and Brax
from datetime import datetime
import functools
import os
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp
import jax.numpy as jnp
import argparse
from pathlib import Path

#@title Import The Playground
#for A100
jax.config.update('jax_default_matmul_precision', 'highest')
from mujoco_playground._src import wrapper
from mujoco_playground._src import registry

from mujoco_playground import locomotion

import robots
from robots.randomize import domain_randomize
from robots.randomize_simple import domain_randomize as domain_randomize_simple

x_data, y_data, y_dataerr = [], [], []
linvels=[]
angvels=[]
times = [datetime.now()]


def progress_training(num_steps, metrics,env_cfg):
  times.append(datetime.now())
  x_data.append(num_steps)

  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])

  print(y_data[-1])
  vel_tracking_per=metrics["eval/episode_reward/tracking_lin_vel"]/(env_cfg.reward_config.scales.tracking_lin_vel* env_cfg.episode_length)
  ang_tracking_per=metrics["eval/episode_reward/tracking_ang_vel"]/(env_cfg.reward_config.scales.tracking_ang_vel* env_cfg.episode_length)
  average_steps=metrics["eval/avg_episode_length"]
  linvels.append(vel_tracking_per)
  angvels.append(ang_tracking_per)
  print("Lin vel",vel_tracking_per,)
  print("ang vel",ang_tracking_per)

  # termination critiria
  if len(y_data)>=2:
    if vel_tracking_per>env_cfg.vel_percentage and ang_tracking_per>env_cfg.vel_percentage and abs((y_data[-1]-y_data[-2])/y_data[-1])<=0.01:
      return True
    elif abs((y_data[-1]-y_data[-2])/y_data[-1])<=0.003:
      return True
  return False


def run_training(args,progress_fn):
  robot_cfg = robots.get_robot_config(args.robot)
  task_name = args.task_name
  terrain_file = args.terrain_file
  checkpoint_folder = args.checkpoint_folder

  env_name = robot_cfg.ROBOT_NAME

  if args.method=="pgtt":
    from robots.joystick_pgtt import Joystick
  elif args.method=="baseline":
    from robots.joystick import Joystick
  elif args.method=="wild":
    from robots.joystick_wild import Joystick

  env_call=functools.partial(Joystick,consts=robot_cfg,task=task_name)
  locomotion.register_environment(env_name,env_call,robot_cfg.default_config)

  env_cfg = registry.get_default_config(env_name)
  if args.method=="pgtt":
    env_cfg=robot_cfg.default_config()
  elif args.method=="baseline":
    env_cfg=robot_cfg.baseline_config()
  elif args.method=="wild":
    env_cfg=robot_cfg.wild_config()

  if args.eval_flag:
      env_cfg.command_config.u_max=[0.7,0.7,0.7]
      env_cfg.command_config.u_min=[-0.7,-0.7,-0.7]
      env_cfg.pert_config.enable=True
  else:
      env_cfg.command_config.u_max=[1.0,1.0,1.0]
      env_cfg.command_config.u_min=[-1.0,-1.0,-1.0]

  env_cfg.gait_freq=robot_cfg.TRAINING_DEFAULTS["gait_freq_override"]
  env = registry.load(env_name,config=env_cfg)
  print(env.mjx_model.nbody)
  print(env.action_size)


  rl_config = config_dict.create(
        num_timesteps=args.num_timesteps,
        num_evals=args.num_evals,
        reward_scaling=1.0,
        episode_length=env_cfg.episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=8,
        discounting=args.discount,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        max_grad_norm=1.0,
        num_eval_envs=args.num_eval_envs,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
    )


  ppo_params=rl_config
  if task_name=="stairs":
    matrix=jnp.load(args.terrain_file)
    randomization=functools.partial(
        domain_randomize,
        terrain_matrix=matrix,
        body_id_offset=robot_cfg.BODY_ID_OFFSET,
        geom_id_offset=robot_cfg.GEOM_ID_OFFSET,
    )
    locomotion._randomizer[env_name]=randomization
  else:
    randomization=functools.partial(
        domain_randomize_simple,
    )
    locomotion._randomizer[env_name]=randomization



  ckpt_path = epath.Path(f"checks_stairs/checkpoint_{args.index}").resolve()
  ckpt_path.mkdir(parents=True, exist_ok=True)
  print(f"{ckpt_path}")

  with open(ckpt_path / "config.json", "w") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)



  progress= functools.partial(progress_fn, env_cfg=env_cfg)


  def policy_params_fn(current_step, make_policy, params):
    del make_policy  # Unused.
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = ckpt_path / f"{current_step}"
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)
    model.save_params(f"policies/policy_{args.index}",params)


  randomizer = registry.get_domain_randomizer(env_name)

  ppo_training_params = dict(ppo_params)
  network_factory = ppo_networks.make_ppo_networks
  if "network_factory" in ppo_params:
    del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory
    )
  train_fn = functools.partial(
      ppo.train, **dict(ppo_training_params),
      network_factory=network_factory,
      randomization_fn=randomizer,
      progress_fn=progress,
      policy_params_fn=policy_params_fn,
  )
  if checkpoint_folder is not None:
    checkpoint_path = Path(f"./{checkpoint_folder}/{get_max_numbered_folder(checkpoint_folder)}").resolve()
    print(f"Restoring from checkpoint: {checkpoint_path}")
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=registry.load(env_name, config=env_cfg),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        restore_checkpoint_path=str(checkpoint_path),
    )
  else:
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=registry.load(env_name, config=env_cfg),
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
  print(f"time to jit: {times[1] - times[0]}")
  print(f"time to train: {times[-1] - times[1]}")
  os.makedirs(f"./plots/{args.method}", exist_ok=True)
  model.save_params(f"policies/policy_{args.index}",params)
  np.save(f"./plots/{args.method}/mean{args.index}",y_data)
  np.save(f"./plots/{args.method}/std{args.index}",y_dataerr)
  np.save(f"./plots/{args.method}/lin_vel{args.index}",linvels)
  np.save(f"./plots/{args.method}/ang_vel{args.index}",angvels)


def get_max_numbered_folder(path):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    number_folders = []

    for f in folders:
        try:
            number = int(f)
            number_folders.append(number)
        except ValueError:
            pass  # skip non-numeric folder names

    return max(number_folders) if number_folders else None
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO with MuJoCo Playground")
    parser.add_argument('--robot', type=str, default='go2', help='Robot name: go2 or anymal')
    parser.add_argument('--method', type=str, default='pgtt', help='pgtt, baseline or wild')
    parser.add_argument('--task_name', type=str, default='stairs', help='Task name: stairs or flat_terrain')
    parser.add_argument('--terrain_file', type=str, default='terrains/level04.npy', help='Path to terrain file')
    parser.add_argument('--checkpoint_folder', type=str, default=None, help='Checkpoint folder to restore from')
    parser.add_argument('--num_envs', type=int, default=None, help='Number of parallel environments (default: from robot config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for PPO (default: from robot config)')
    parser.add_argument('--discount', type=float, default=0.97, help='Discount factor')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--entropy_cost', type=float, default=0.01, help='Entropy cost')
    parser.add_argument('--num_minibatches', type=int, default=None, help='Number of minibatches (default: from robot config)')
    parser.add_argument('--num_timesteps', type=int, default=200_000_000, help='Total number of timesteps')
    parser.add_argument('--num_evals', type=int, default=31, help='Number of evaluations')
    parser.add_argument('--index', type=str, default='0', help='Name/identifier for checkpoint saving')
    parser.add_argument('--num_eval_envs', type=int, default=128, help='Number of evaluation environments')
    parser.add_argument('--eval_flag', type=bool, default=False, help='True if you want to evaluate the model')
    args = parser.parse_args()

    # Fill in defaults from robot config when not explicitly provided
    robot_cfg = robots.get_robot_config(args.robot)
    defaults = robot_cfg.TRAINING_DEFAULTS
    if args.num_envs is None:
        args.num_envs = defaults["num_envs"]
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]
    if args.num_minibatches is None:
        args.num_minibatches = defaults["num_minibatches"]

    run_training(args,progress_training)
