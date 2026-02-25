#!/usr/bin/env python3
"""
Visualise a single environment from a pre-generated terrain .npy file.

The .npy matrix has shape (num_envs, num_bodies, 10):
  cols 0-2  : body position (x, y, z)
  cols 3-6  : body quaternion (w, x, y, z)
  cols 7-9  : geom half-sizes (sx, sy, sz)

The body/geom offsets match those used in robots/randomize.py:
  body_id_offset = 14   (world body + 13 go2 bodies)
  geom_id_offset = 43   (floor + all go2 geoms)

Usage examples
--------------
# default: first env from terrains/level.npy with go2
python test_terrain_viewer.py

# pick a specific env
python test_terrain_viewer.py --env_idx 5

# random env
python test_terrain_viewer.py --env_idx -1

# different level file
python test_terrain_viewer.py --file terrains/level05.npy

# anymal robot (different offsets – check with --detect)
python test_terrain_viewer.py --robot anymal --detect
"""
import argparse
import os
import numpy as np
import mujoco
import mujoco.viewer


def find_box_ids(model: mujoco.MjModel):
    """Return (body_ids, geom_ids) for bodies whose name starts with 'box_'."""
    body_ids, geom_ids = [], []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith("box_"):
            body_ids.append(i)
    for i in range(model.ngeom):
        # geom parent body
        bid = model.geom_bodyid[i]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name and name.startswith("box_"):
            geom_ids.append(i)
    return np.array(body_ids), np.array(geom_ids)


def main():
    parser = argparse.ArgumentParser(description="View terrain from .npy file in MuJoCo")
    parser.add_argument("--file", type=str, default="terrains/level.npy",
                        help="Terrain .npy file to load (default: terrains/level.npy)")
    parser.add_argument("--env_idx", type=int, default=0,
                        help="Environment index to show (default: 0, -1 = random)")
    parser.add_argument("--robot", type=str, default="go2",
                        help="Robot name: go2 or anymal (default: go2)")
    parser.add_argument("--body_offset", type=int, default=14,
                        help="Body ID offset for terrain boxes (default: 14)")
    parser.add_argument("--geom_offset", type=int, default=43,
                        help="Geom ID offset for terrain boxes (default: 43)")
    parser.add_argument("--detect", action="store_true",
                        help="Auto-detect body/geom offsets from box_ names instead of using offsets")
    args = parser.parse_args()

    # ── Load terrain matrix ──────────────────────────────────────────────────
    if not os.path.isfile(args.file):
        raise FileNotFoundError(f"Terrain file not found: {args.file}")

    terrain = np.load(args.file)
    num_envs, num_bodies, _ = terrain.shape
    print(f"Terrain: {args.file}")
    print(f"  shape = {terrain.shape}  (num_envs={num_envs}, num_bodies={num_bodies})")

    env_idx = np.random.randint(num_envs) if args.env_idx == -1 else args.env_idx % num_envs
    print(f"  Visualising env_idx = {env_idx}")
    env_data = terrain[env_idx]  # (num_bodies, 10)

    # ── Load MuJoCo model ───────────────────────────────────────────────────
    xml_path = os.path.join(args.robot, "xmls", "terrain_scene_mjx.xml")
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"Scene XML not found: {xml_path}")

    print(f"Model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # ── Resolve body / geom IDs ─────────────────────────────────────────────
    if args.detect:
        body_ids, geom_ids = find_box_ids(model)
        print(f"  Auto-detected {len(body_ids)} box bodies, {len(geom_ids)} box geoms")
        if len(body_ids) == 0:
            raise RuntimeError("No bodies named 'box_*' found in model.")
    else:
        body_ids = np.arange(args.body_offset, args.body_offset + num_bodies)
        geom_ids = np.arange(args.geom_offset, args.geom_offset + num_bodies)
        print(f"  Using body ids [{body_ids[0]}..{body_ids[-1]}], "
              f"geom ids [{geom_ids[0]}..{geom_ids[-1]}]")

    n = min(num_bodies, len(body_ids), len(geom_ids))

    # ── Apply terrain data to model ─────────────────────────────────────────
    model.body_pos[body_ids[:n]]  = env_data[:n, :3]
    model.body_quat[body_ids[:n]] = env_data[:n, 3:7]
    model.geom_size[geom_ids[:n]] = env_data[:n, 7:]

    mujoco.mj_forward(model, data)

    # ── Launch viewer ────────────────────────────────────────────────────────
    print("Launching MuJoCo viewer  (close window to exit)")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
