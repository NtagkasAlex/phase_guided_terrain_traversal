import mujoco
import numpy as np
from mujoco import mjx
import jax
import jax.numpy as jnp
from functools import partial
from go2.go2_constants import *


@jax.jit
def raycast_sensor(mjx_model,mjx_data, pos):

	ray_sensor_site = jnp.array([pos[0], pos[1], pos[2]])
	direction_vector = jnp.array([0, 0, -1.])
	geomgroup_mask=(1,0,0,0,1,1)

	f_ray=partial(mjx.ray,vec=direction_vector,
				geomgroup=geomgroup_mask)
	
	z=f_ray(mjx_model,mjx_data,ray_sensor_site)
	intersection_point = ray_sensor_site + direction_vector * z[0]
	
	return intersection_point

@jax.jit
def create_sensor_matrix(mx:mjx.Model,dx:mjx.Data,center, yaw=0., key=None, noise_range=0.005):
		"""
		This is the main function used to create the grid map using the ray sensor data
		"""

		R_W2H = jnp.array([jnp.cos(yaw), jnp.sin(yaw), -jnp.sin(yaw), jnp.cos(yaw)])
		R_W2H = R_W2H.reshape((2, 2))

		c_h = (num_heightscans - 1) / 2
		c_w = (num_widthscans - 1) / 2

		ref_robot = jnp.array([center[0], center[1], center[2] + 0.6])

		idx_h = jnp.arange(num_heightscans)
		idx_w = jnp.arange(num_widthscans)

		p, k = jnp.meshgrid(c_h - idx_h, c_w - idx_w, indexing="ij")
		offsets = jnp.stack([p * dist_x, k * dist_y], axis=-1)  # (H, W, 2)

		if key is not None:
			key, subkey = jax.random.split(key)
			noise = jax.random.uniform(subkey, shape=offsets.shape, minval=-noise_range, maxval=noise_range)
			offsets = offsets + noise

		offsets = offsets @ R_W2H  

		grid_positions = jnp.concatenate([
			ref_robot[:2] + offsets,
			jnp.full((num_heightscans, num_widthscans, 1), ref_robot[2])
		], axis=-1)

		center_row = jnp.round(c_h).astype(jnp.int32)
		center_col = jnp.round(c_w).astype(jnp.int32)

		sensor_matrix = grid_positions.at[center_row, center_col].set(ref_robot)

		get_data = jax.vmap(
			jax.vmap(raycast_sensor, in_axes=(None, None, 0)),
			in_axes=(None, None, 0)
		)(mx, dx, sensor_matrix)

		return get_data