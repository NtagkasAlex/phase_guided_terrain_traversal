import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
@jax.jit
def quat_to_yaw(quat):
    quat_scipy = jnp.roll(quat, shift=-1)# mujoco -> scipy convention
    yaw = Rotation.from_quat(quat_scipy).as_euler('xyz')[2]
    return yaw
