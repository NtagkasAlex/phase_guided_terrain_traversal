import mujoco
import numpy as np
from functools import partial
from go2.go2_constants import *

def raycast_sensor(model, data, pos):
    ray_sensor_site = np.array([pos[0], pos[1], pos[2]])
    direction_vector = np.array([0, 0, -1.])
    geomgroup_mask =np.array([1, 0, 0, 0, 1, 1], dtype=np.uint8)

    flg_static = 1  
    bodyexclude = (
        -1
    )  
    geomid = np.zeros(1, dtype=np.int32)

    z = mujoco.mj_ray(model, data, ray_sensor_site, direction_vector, geomgroup=geomgroup_mask,flg_static=flg_static,bodyexclude=bodyexclude,geomid=geomid)
    intersection_point = ray_sensor_site + direction_vector * z


    return intersection_point#-(data.qpos[2]-0.28)
def create_sensor_matrix(model, data, center, yaw=0.,noise_xy=0.00,noise_z=0.0):
    """
    This function creates the grid map using ray sensor data.
    
    Args:
        model: The simulation/model object.
        data: The simulation data.
        center: Tuple of (x, y, z) center position.
        num_heightscans: Number of scans in the height (rows) direction.
        num_widthscans: Number of scans in the width (columns) direction.
        dist_x: Distance between grid points along the x-axis.
        dist_y: Distance between grid points along the y-axis.
        yaw: Rotation (in radians) to apply to the grid (default 0).
    
    Returns:
        A (num_heightscans x num_widthscans x 3) matrix with sensor data.
    """
    R_W2H = np.array([
        [np.cos(yaw), np.sin(yaw)],
        [-np.sin(yaw), np.cos(yaw)]
    ])

    c_h = (num_heightscans - 1) / 2  # center row
    c_w = (num_widthscans - 1) / 2   # center column

    ref_robot = np.array([center[0], center[1], center[2] + 0.6])

    idx_h = np.arange(num_heightscans)
    idx_w = np.arange(num_widthscans)

    p, k = np.meshgrid(c_h - idx_h, c_w - idx_w, indexing="ij")
    offsets = np.stack([p * dist_x, k * dist_y], axis=-1)  # (H, W, 2)
    offsets += np.random.uniform(-noise_xy, noise_xy, size=offsets.shape)

    offsets = offsets @ R_W2H  # rotate

    grid_positions = np.concatenate([
        ref_robot[:2] + offsets,
        np.full((num_heightscans, num_widthscans, 1), ref_robot[2])
    ], axis=-1)

    sensor_matrix = grid_positions.copy()

    center_row = int(np.round(c_h))
    center_col = int(np.round(c_w))
    sensor_matrix[center_row, center_col] = ref_robot  # ensure center is exact

    def noisy_raycast(pos):
        ray = raycast_sensor(model, data, pos)
        noise = np.random.uniform(-noise_z, noise_z, size=1)
        return ray + noise

    get_data = np.apply_along_axis(noisy_raycast, -1, sensor_matrix)


    return get_data
