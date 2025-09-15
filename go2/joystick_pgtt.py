from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
import go2.base as go2_base
import go2.go2_constants as consts
import go2.gait as gait
from go2.heightmap import create_sensor_matrix
import go2.joystick_base as joystick_base
from go2.configs import default_config

class Joystick(joystick_base.Joystick_Base):
  """Track a joystick command."""
  def __init__( 
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        task=task,
        config=config,
        config_overrides=config_overrides,
    )