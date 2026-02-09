from typing import Any, Dict, Optional, Union

from ml_collections import config_dict

import robots.joystick_base as joystick_base


class Joystick(joystick_base.Joystick_Base):
  """PGTT (Phase-Guided Terrain Traversal) joystick variant."""
  def __init__(
      self,
      consts,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = None,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    if config is None:
      config = consts.default_config()
    super().__init__(
        consts=consts,
        task=task,
        config=config,
        config_overrides=config_overrides,
    )
