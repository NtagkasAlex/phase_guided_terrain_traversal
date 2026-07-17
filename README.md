# PGTT: Phase-Guided Terrain Traversal for Perceptive Legged Locomotion
<p align="center">
<img src="docs/img/real.jpg" alt="Example Gait" width="70%" />
</p>

A perceptive reinforcement learning locomotion framework developed for the Unitree GO2 and ANYmal in simulation (MuJoCo MJX). Deployed on real hardware using unitree sdk2py for robot control and Point-Lio and Elevation Mapping for the perception pipeline, using the Unitree L1 LiDAR.
<p align="center">
<img src="docs/img/sim.jpg" alt="Example Gait" width="70%" />
</p>

## Project Structure

```
.
├── robots/                  # Shared robot-agnostic code
│   ├── __init__.py          # Robot registry: robots.get_robot_config("go2")
│   ├── base.py              # RobotEnv base class (parameterized by consts)
│   ├── joystick_base.py     # Shared Joystick_Base with all reward functions
│   ├── joystick_pgtt.py     # PGTT joystick variant
│   ├── joystick.py          # Baseline joystick variant
│   ├── joystick_wild.py     # Wild joystick variant (oscillator pose)
│   ├── gait.py              # Gait controller (spline-based trajectories)
│   ├── heightmap.py         # Heightmap sensor (JAX, parameterized grid spacing)
│   ├── randomize.py         # Domain randomization (parameterized offsets)
│   ├── randomize_simple.py  # Simple domain randomization
│   └── utility.py           # Shared utilities
├── go2/
│   ├── robot_config.py      # Go2-specific constants, configs, and defaults
│   └── xmls/                # Go2 MuJoCo XML scene files
├── anymal/
│   ├── robot_config.py      # ANYmal-specific constants, configs, and defaults
│   └── xmls/                # ANYmal MuJoCo XML scene files
├── training/
│   └── train.py             # Unified training script (--robot flag)
├── deploy/
│   ├── deploy_heightmap.py  # MuJoCo simulation deployment (--robot flag)
│   ├── deploy_real.py       # Real hardware deployment (Unitree SDK)
│   ├── cpu_heightmap/       # CPU-based heightmap for deployment
│   └── policy_net.py        # Policy network loader (PyTorch)
├── terrain/
│   └── generator.py         # Terrain generation via WFC (--robot flag)
└── policies/                # Saved policy checkpoints
```

All shared logic lives in `robots/`. Robot-specific parameters (XML paths, PD gains, heightmap spacing, reward scales, training hyperparameters) are defined in `go2/robot_config.py` and `anymal/robot_config.py`. Every script accepts a `--robot` flag to select which robot to use.

## Terrain Generation
Terrains are produced using Wave Function Collapse in MuJoCo.

<p align="center">
  <img src="docs/img/example1.png" alt="Example 1" width="45%" />
  <img src="docs/img/example2.png" alt="Example 2" width="45%" />
</p>

To generate terrains:
```bash
# Generate level .npy files for training (Go2 default)
python terrain/generator.py levels

# Generate level .npy files for ANYmal
python terrain/generator.py --robot anymal levels

# Fill template XML with placeholder boxes
python terrain/generator.py fill --num_objects 100

# Create a random test terrain for visualization
python terrain/generator.py test --step_height 0.08 --width 0.4 --num_steps 4
```

See [terrain/README.md](terrain/README.md) for the full argument reference for each subcommand.

## Training Pipeline

<p align="center">
<img src="docs/img/main.jpg" alt="Main Figure" width="100%" />
</p>

## Real World Experiment
The resulting policy deployed in the real world. We use Point-Lio for odometry and Gridmap to extract the desired heightmap.


https://github.com/user-attachments/assets/06f6cd29-2e20-4c2a-a6c7-c1781c9743b1



## Installation
Create a conda environment (recommended):
```bash
conda create -n pgtt python=3.12 -y
conda activate pgtt
```

Install JAX for GPU for your CUDA version.
To find your CUDA version:
```bash
nvidia-smi
```

```bash
pip install "jax[cuda12]==0.8.0"
```
if CUDA version is 13.

Install the required dependencies:p

```bash
pip install -r requirements.txt
```

## Training

### Quick start
```bash
# Train Go2 with PGTT method on stairs
python training/train.py --robot go2 --method pgtt --task_name stairs

# Train ANYmal with PGTT method on stairs
python training/train.py --robot anymal --method pgtt --task_name stairs
```

### Available methods
| Method     | Description                                         |
|------------|-----------------------------------------------------|
| `pgtt`     | Phase-Guided Terrain Traversal (default)            |
| `baseline` | Baseline without phase/gait frequency in obs        |
| `wild`     | Uses oscillator pose from gait trajectory generator |

### Training arguments
| Argument              | Default              | Description                                      |
|-----------------------|----------------------|--------------------------------------------------|
| `--robot`             | `go2`                | Robot: `go2` or `anymal`                         |
| `--method`            | `pgtt`               | Training method: `pgtt`, `baseline`, or `wild`   |
| `--task_name`         | `stairs`             | Task: `stairs` or `flat_terrain`                 |
| `--terrain_file`      | `terrains/level04.npy` | Path to terrain heightmap file                 |
| `--num_envs`          | from robot config    | Number of parallel environments                  |
| `--batch_size`        | from robot config    | PPO batch size                                   |
| `--num_minibatches`   | from robot config    | Number of PPO minibatches                        |
| `--num_timesteps`     | `200_000_000`        | Total training timesteps                         |
| `--learning_rate`     | `3e-4`               | Learning rate                                    |
| `--discount`          | `0.97`               | Discount factor                                  |
| `--num_evals`         | `31`                 | Number of evaluations during training            |
| `--index`             | `0`                  | Identifier for checkpoint saving                 |
| `--checkpoint_folder` | `None`               | Resume training from checkpoint                  |
| `--eval_flag`         | `False`              | Enable evaluation mode (restricted commands)     |

Default training hyperparameters (num_envs, batch_size, num_minibatches, gait_freq) are loaded from each robot's config and can be overridden via CLI.

### Examples
```bash
# Train Go2 baseline on flat terrain
python training/train.py --robot go2 --method baseline --task_name flat_terrain

# Train ANYmal PGTT with custom hyperparameters
python training/train.py --robot anymal --method pgtt --num_envs 4096 --batch_size 1024

# Resume training from checkpoint
python training/train.py --robot go2 --method pgtt --checkpoint_folder checks_stairs/checkpoint_0
```

### Full training pipeline (training.sh)
The main training script runs all methods across multiple terrain difficulty levels with curriculum learning. It takes the robot name as the first argument:
```bash
# Train Go2 (default)
bash training/training.sh go2

# Train ANYmal
bash training/training.sh anymal
```
This runs 5 independent runs, each training `pgtt`, `baseline`, and `wild` methods across levels `level03 → level07 → level10 → level13`, using curriculum learning (each level resumes from the previous checkpoint).

## Evaluation

Evaluate trained checkpoints on discrete terrains and/or stair heights:
```bash
# Run both discrete and stair evaluations for Go2
python training/evaluate_multiple.py --robot go2 --eval_type both

# Only discrete terrain evaluation for ANYmal
python training/evaluate_multiple.py --robot anymal --eval_type discrete

# Only stair height evaluation with custom height range
python training/evaluate_multiple.py --robot go2 --eval_type stairs --height_min 1 --height_max 10

# Evaluate a specific training run
python training/evaluate_multiple.py --robot go2 --eval_type both --run 2
```

| Argument       | Default | Description                                      |
|----------------|---------|--------------------------------------------------|
| `--robot`      | `go2`   | Robot: `go2` or `anymal`                         |
| `--eval_type`  | `both`  | Evaluation type: `stairs`, `discrete`, or `both` |
| `--run`        | `0`     | Which training run to evaluate                   |
| `--height_min` | `1`     | Min stair height in cm (for stairs eval)         |
| `--height_max` | `10`    | Max stair height in cm (for stairs eval)         |

Results are saved as `.npy` files in `plots/`.

## Deployment

### Simulation deployment (MuJoCo viewer)
```bash
# Deploy Go2 policy in simulation
python deploy/deploy_heightmap.py --robot go2

# Deploy ANYmal policy in simulation
python deploy/deploy_heightmap.py --robot anymal

# Enable or disable stairs terrain
python deploy/deploy_heightmap.py --stairs
python deploy/deploy_heightmap.py --no-stairs

# Override command velocity
python deploy/deploy_heightmap.py --vx 1.0 --vy 0.0 --yaw 0.0
```

Default deployment parameters (command velocity, gait frequency, policy path, perturbation settings) are loaded from `DEPLOY_DEFAULTS` in each robot's config. CLI flags override the config values; any flag that is omitted falls back to the config default.

| Argument    | Default         | Description                                      |
|-------------|-----------------|--------------------------------------------------|
| `--robot`   | `go2`           | Robot: `go2` or `anymal`                         |
| `--method`  | `pgtt`          | Method: `pgtt`, `baseline`, or `wild`            |
| `--level`   | `level03`       | Terrain level used to select the policy          |
| `--run`     | `0`             | Policy run index                                 |
| `--stairs`  / `--no-stairs` | from config | Enable or disable stair terrain      |
| `--vx`      | from config     | Forward speed command (m/s)                      |
| `--vy`      | from config     | Lateral speed command (m/s)                      |
| `--yaw`     | from config     | Yaw rate command (rad/s)                         |

### Real hardware deployment

Two terminals are required:

**Terminal 1 — Perception pipeline (Docker)**

Make sure `/utlidar/cloud` and `/utlidar/imu` topics are providing data, then start the elevation mapping container (see [Perception pipeline](#perception-pipeline-using-docker)):
```bash
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  --net=host \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  -e MESA_LOADER_DRIVER_OVERRIDE=llvmpipe \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  ros2-humble-dev
```
Inside the container(only when you actually want to start the heightmap and the robot is in stand mode):
```bash
cd ros_scripts
chmod +x run_elevation.sh
./run_elevation.sh
```
An RViz window should appear showing the gridmap and scan points. Leave this running.

**Terminal 2 — Policy (conda env `pgtt`)**

```bash
conda activate pgtt

# Deploy with controller (joystick) commands (default)
python -m deploy.deploy_real --robot go2 --method pgtt --level level03 --run 0

# Deploy with fixed velocity commands
python -m deploy.deploy_real --robot go2 --method pgtt --level level03 --run 0 \
    --command_type fixed --vx 0.2 --vy 0.0 --yaw 0.0
```

| Argument         | Default        | Description                                              |
|------------------|----------------|----------------------------------------------------------|
| `--robot`        | `go2`          | Robot: `go2` or `anymal`                                 |
| `--method`       | `pgtt`         | Method: `pgtt`, `baseline`, or `wild`                    |
| `--level`        | `level03`      | Terrain level used to select the policy checkpoint       |
| `--run`          | `0`            | Policy run index                                         |
| `--command_type` | `controller`   | Command source: `controller` (joystick) or `fixed`       |
| `--vx`           | `0.2`          | Forward speed when using `fixed` command (m/s)           |
| `--vy`           | `0.0`          | Lateral speed when using `fixed` command (m/s)           |
| `--yaw`          | `0.0`          | Yaw rate when using `fixed` command (rad/s)              |

**Startup sequence:**
1. Press **Enter** in Terminal 2 to confirm and begin.
2. The robot automatically stands up over ~7 seconds.
3. Once fully standing, the terminal prints `"Press B to start policy execution."`.Switch to Terminal 1 (Docker) and run `./run_elevation.sh`. Wait for RViz to show a stable, drift-free heightmap with correct ground estimation.
4. Press **B** on the controller to start the policy.
5. Press **A** at any time to immediately stop the policy and sit the robot down safely.

## Adding a New Robot

1. Create a directory `<robot_name>/` with `xmls/` containing your MuJoCo scene files.
2. Create `<robot_name>/robot_config.py` following the structure of `go2/robot_config.py`:
   - Define XML paths, sensor names, heightmap spacing, PD gains
   - Define `default_config()`, `baseline_config()`, `wild_config()` functions
   - Define `TRAINING_DEFAULTS` and `DEPLOY_DEFAULTS` dicts
3. Register the robot in `robots/__init__.py`:
   ```python
   _REGISTRY = {
       "go2": "go2.robot_config",
       "anymal": "anymal.robot_config",
       "your_robot": "your_robot.robot_config",
   }
   ```
4. All training, deployment, and terrain scripts will work with `--robot your_robot`.

## Perception pipeline using docker
Build the image once:
```bash
docker build -t ros2-humble-dev .
```
Running instructions are part of the [Real hardware deployment](#real-hardware-deployment) two-terminal setup above.

## Citation
If you find our work useful in your research, please cite it as follows:
```bash

@inproceedings{ntagkas2025pgtt,
  title={PGTT: Phase-Guided Terrain Traversal for Perceptive Legged Locomotion},
  author={Ntagkas, Alexandros and Kiourt, Chairi and Chatzilygeroudis, Konstantinos},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2026}
}
```
