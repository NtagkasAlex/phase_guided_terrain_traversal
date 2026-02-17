An essential part of our approach is terrain generation in MuJoCo, which uses several hyperparameters to create the desired terrain.

## Usage

### 1. Generate level terrain files (.npy)

Produces `terrains/level01.npy` through `terrains/level09.npy` and `terrains/discrete.npy` for domain randomization during training.

```bash
python terrain/generator.py --robot anymal levels
```

| Argument         | Default | Description                                      |
|------------------|---------|--------------------------------------------------|
| `--num_envs`     | `100`   | Number of parallel terrain environments           |
| `--num_objects`  | `100`   | Max scene objects per environment                 |
| `--size`         | `9`     | Size of the terrain grid                          |
| `--num_steps`    | `3`     | Number of steps in a single stair                 |
| `--width`        | `0.1`   | Width of steps                                    |
| `--step_height`  | `0.15`  | Height of steps                                   |

### 2. Fill template scene with placeholder boxes

Writes `terrain_scene_mjx.xml` with `num_objects` placeholder boxes used during training (repositioned at runtime via domain randomization).

```bash
python terrain/generator.py --robot anymal fill --num_objects 100
```

| Argument         | Default | Description                                      |
|------------------|---------|--------------------------------------------------|
| `--num_objects`  | `100`   | Number of placeholder boxes                       |
| `--num_steps`    | `3`     | Number of steps in a single stair                 |
| `--width`        | `0.1`   | Width of steps                                    |
| `--step_height`  | `0.15`  | Height of steps                                   |

### 3. Create random test terrain

Generates a random stair terrain in `terrain_test_mjx.xml` for evaluation and visualization.

```bash
python terrain/generator.py --robot anymal test --num_objects 100
```

| Argument         | Default | Description                                      |
|------------------|---------|--------------------------------------------------|
| `--num_objects`  | `100`   | Max scene objects (unused slots become placeholders) |
| `--size`         | `9`     | Size of the terrain grid                          |

## How it works

For each robot we have to create 2 additional xml's (`terrain_scene_mjx.xml` and `terrain_test_mjx.xml`).

The first contains `num_objects` boxes that are used as placeholders and are moved on domain randomization based on the desired terrain type and shape.

The second is used for evaluation, where we produce a single scene in which we can visualize the trained policy.
