An essential part of our approach is terrain generation in MuJoCo, which uses several hyperparameters to create the desired terrain.


| Argument                  | Default                 | Description                                                                 |
|------------------------|--------------------------|-----------------------------------------------------------------------------|
| `size`                | `5`               | Size of the terrain grdi                          |
| `length`            | `None`              | Length of a single stair step|
| `num_steps`         | `4` | Number of steps in a single stair                             |
| `width`         | `0.4` | Width of steps |
| `step_height`         | `0.1` | Height of steps                            |
| `num_envs`         | `100` | Number of different terrains that are used for parallel training|
| `num_objects`         | `100` | Number of maximum scene objects                            |

For each robot we have to create 2 additional xml's ( terrain_scene_mjx.xml and terrain_test_mjx.xml).

The first contains `num_objects` boxes that are used as placeholders and are moved on domain randomization based on the desired terrain type and shape.

The second is used for evaluation, where we produce a single scene in which we can visualize the trained policy.
