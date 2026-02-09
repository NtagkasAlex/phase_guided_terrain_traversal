"""Robot registry for the PGTT project.

Usage:
    import robots
    cfg = robots.get_robot_config("go2")  # or "anymal"
    cfg.default_config()
    cfg.task_to_xml("stairs")
    cfg.dist_x, cfg.dist_y, ...
"""

import importlib

_REGISTRY = {
    "go2": "go2.robot_config",
    "anymal": "anymal.robot_config",
}


def get_robot_config(name):
    """Load and return the robot config module by name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown robot: {name!r}. Available: {list(_REGISTRY.keys())}"
        )
    return importlib.import_module(_REGISTRY[name])


def available_robots():
    """Return list of registered robot names."""
    return list(_REGISTRY.keys())
