import mujoco
import numpy as np
from functools import partial
from go2.go2_constants import *

def raycast_sensor(model, data, pos):
    ray_sensor_site = np.array([pos[0], pos[1], pos[2]])
    direction_vector = np.array([0, 0, -1.])
    geomgroup_mask =np.array([1, 0, 0, 0, 1, 1], dtype=np.uint8)

    flg_static = 1  # A flag indicating whether the ray should only intersect with static objects (1) or also with dynamic objects (0)
    bodyexclude = (
        -1
    )  # An optional parameter specifying a body ID to exclude from intersections. Use -1 to not exclude any body.
    geomid = np.zeros(1, dtype=np.int32)

    z = mujoco.mj_ray(model, data, ray_sensor_site, direction_vector, geomgroup=geomgroup_mask,flg_static=flg_static,bodyexclude=bodyexclude,geomid=geomid)
    intersection_point = ray_sensor_site + direction_vector * z


    return intersection_point#-(data.qpos[2]-0.28)

# def create_sensor_matrix(model, data, center, yaw=0.):
#     """
#     This function creates the grid map using ray sensor data.
#     """
#     R_W2H = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
#     c = (num_heightscans - 1) // 2

#     ref_robot = np.array([center[0], center[1], center[2] + 0.6])
#     idx = np.arange(num_heightscans)
#     # p, k = np.meshgrid(idx - c, idx - c, indexing="ij")
#     p, k = np.meshgrid(c-idx,c-idx, indexing="ij")
#     offsets = np.stack([p * dist_x, k * dist_y], axis=-1)  # (N, N, 2)
#     # noise_level = 0.02  # Adjust based on your desired noise strength
#     # noise = np.random.uniform(low=-noise_level, high=noise_level, size=offsets.shape)
#     # offsets+=noise
#     offsets = offsets @ R_W2H

#     grid_positions = np.concatenate([
#         ref_robot[:2] + offsets,
#         np.full((num_heightscans, num_heightscans, 1), ref_robot[2])  
#     ], axis=-1)

#     sensor_matrix = grid_positions.copy()
#     sensor_matrix[c, c] = ref_robot  # (N, N, 3)
#     # sensor_matrix=sensor_matrix.reshape((-1,3))
#     # get_data = np.array([raycast_sensor(model, data, pos) for pos in sensor_matrix])
#     # get_data=get_data.reshape((num_heightscans,num_heightscans,3))
#     get_data = np.apply_along_axis(lambda pos: raycast_sensor(model, data, pos), -1, sensor_matrix)

#     return get_data

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

num_points=9
radius=0.1
def create_feet_heightscan(model, data, center):
    """
    Create a circular perimeter of ray sensor data points around the robot.
    
    Parameters:
    - model, data: used in the raycast_sensor function.
    - center: (x, y, z) position of the robot.
    - yaw: orientation of the robot in radians.
    - num_points: number of points on the perimeter.
    - radius: distance from the center to each sensor point.
    
    Returns:
    - get_data: array of raycast_sensor outputs for each perimeter point.
    """
   
    # Reference robot position with height offset
    ref_robot = np.array([center[0], center[1], center[2] + 0.6])

    # Compute circular perimeter points in local frame
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x_offsets = radius * np.cos(angles)
    y_offsets = radius * np.sin(angles)
    offsets = np.stack([x_offsets, y_offsets], axis=-1)  # (num_points, 2)

    # Apply rotation

    # Compute final positions
    perimeter_positions = np.concatenate([
        ref_robot[:2] + offsets,
        np.full((num_points, 1), ref_robot[2])
    ], axis=-1)

    # Cast rays and gather data
    get_data = np.apply_along_axis(lambda pos: raycast_sensor(model, data, pos), -1, perimeter_positions)

    return get_data