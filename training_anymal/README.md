# Training

Training is performed through the bash file training.sh which calls train.py with different arguements.

## Usage
```bash
python3 train.py --<argument-name> --<value>
```

## Supported Arguments
The following command-line arguments are supported by `train.py`:

| Argument                  | Default                 | Description                                                                 |
|------------------------|--------------------------|-----------------------------------------------------------------------------|
| `--method`                | `'pgtt'`                | Training method: `pgtt`, `baseline`, or `wild`.                            |
| `--task_name`            | `'stairs'`              | Task environment: `stairs` or `flat_terrain`.                              |
| `--terrain_file`         | `'terrains/level05.npy'` | Path to the NumPy file containing terrain data.                            |
| `--checkpoint_folder`   | `None`                  | Path to a checkpoint folder to restore training from.                      |
| `--num_envs`              | `4096`                  | Number of parallel environments.                                           |
| `--batch_size`            | `256`                   | Batch size used in PPO updates.                                            |
| `--discount`            | `0.97`                  | Discount factor for computing returns.                                     |
| `--learning_rate`       | `3e-4`                  | Learning rate for the PPO optimizer.                                       |
| `--num_minibatches`       | `32`                    | Number of minibatches per PPO epoch.                                       |
| `--num_timesteps`         | `100_000_000`           | Total number of training timesteps.                                        |
| `--num_evals`             | `31`                    | Number of evaluation runs during training.                                 |
| `--index`                | `0`                    | Index used to save checkpoints and identify training runs.                 |

Obviously for the task flat_terrain the terrain file has no application.

The training pipeline has 2 important functions that are called at each evaluation.
### Policy Parameters Function
The policy paremeters function is responsible for saving checkpoints and saving a policy for each evaluation. The way I have set it up the policy has the same name as the final policy , but this is not necessary.
**This should be called before the next function (progress function) to save the right policy if/when training termination arises.**
### Progress Function
The progress function can read the metrics we use e.g. rewards and/or total time of the episode for all evaluation agents (takes the mean value of them).

The way I utilize it is to print the metrics of interest like velocity tracking or survival time and/or stop the training when the policy meets some criteria.



## Deployment 

After training a policy should be saved as policy<index>.

To see the results you should change the filename of the policy in file deploy_heightmap.py and run it as a module from 

```
python3 -m deploy.deploy_heightmap
```

## Memory Usage
XLA preallocates memory to avoid the overhead of dynamic allocations later, so decreasing this may reduce performance, as stated in this [issue](https://github.com/google-deepmind/mujoco_playground/issues/102) .

Therefore if you mind using your whole GPU VRAM (well 80% of it) you should add this command towards the start of the train.py file 

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '**p**'

Where **p** is the percentage of desired memory usage and can range from 0 to 1.0.

