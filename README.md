# PGTT: Phase-Guided Terrain Traversal for Perceptive Legged Locomotion
<p align="center">
<img src="docs/img/real.jpg" alt="Example Gait" width="70%" />
</p>

A perceptive reinforcement learning locomotion framework developed for the Unitree GO2 in simulation (MuJoco MJX) and deployed using unitree sdk2py, on real hardware.
<p align="center">
<img src="docs/img/sim.jpg" alt="Example Gait" width="70%" />
</p>

## Terrain Generation
Terrain are produced using Wave Function Collapse in MuJoCo.

<p align="center">
  <img src="docs/img/example1.png" alt="Example 1" width="45%" />
  <img src="docs/img/example2.png" alt="Example 2" width="45%" />
</p>

## Training Pipeline 

<p align="center">
<img src="docs/img/main.jpg" alt="Main Figure" width="100%" />
</p>


## Real World Experiment
The resulting policy deployed in the real world. We use Point-Lio for odometry and Gridmap to extract the desired heightmap.


https://github.com/user-attachments/assets/06f6cd29-2e20-4c2a-a6c7-c1781c9743b1



## Installation
Create a conda environment(recommended)
```bash
conda create -n pgtt python=3.12 -y
conda activate pgtt
```

Install jax for GPU for your cuda version.
To find your cuda version 
```bash
nvidia-smi
```

```bash
pip install -U "jax[cuda_13]"
```
if cuda version is 13

Install the required dependencies:

```bash
pip install -r requirements.txt
```
## Training 
To train the policy run the bash file as:
```bash
bash training/training.sh
```
Additionally, you could modify the hyperparameters from inside this file or the training difficulty.

## Instructions 
For more detailed instructions, please refer to the README file in each folder.
