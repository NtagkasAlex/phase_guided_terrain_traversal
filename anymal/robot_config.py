"""ANYmal robot configuration - all robot-specific parameters in one place."""

from etils import epath
from ml_collections import config_dict

# ---- Identity ----
ROBOT_NAME = "anymal"
ROOT_PATH = epath.Path("anymal")

# ---- XML paths ----
FEET_ONLY_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_feetonly.xml"
FEET_ONLY_FLAT_STAIRS_XML = ROOT_PATH / "xmls" / "terrain_scene_mjx.xml"
TEST_STAIRS_XML = ROOT_PATH / "xmls" / "terrain_test_mjx.xml"
HUGE_STAIRS_XML = ROOT_PATH / "xmls" / "huge_stairs.xml"
FEET_ONLY_ROUGH_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_feetonly_rough_terrain.xml"
FULL_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx.xml"
FULL_COLLISIONS_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_fullcollisions_flat_terrain.xml"


def task_to_xml(task_name: str) -> epath.Path:
    return {
        "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
        "stairs": FEET_ONLY_FLAT_STAIRS_XML,
        "test_stairs": TEST_STAIRS_XML,
        "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
        "huge_stairs": HUGE_STAIRS_XML,
    }[task_name]


# ---- Body/Sensor names ----
FEET_SITES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
FEET_GEOMS = ["FR", "FL", "RR", "RL"]
FEET_POS_SENSOR = ["FR_pos", "FL_pos", "RR_pos", "RL_pos"]
ROOT_BODY = "base"
UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

# ---- Heightmap parameters ----
dist_x = 0.15
dist_y = 0.15
num_heightscans = 11
num_widthscans = 9

# ---- Randomization parameters ----
GEOM_ID_OFFSET = 43
BODY_ID_OFFSET = 14

# ---- Training defaults ----
TRAINING_DEFAULTS = dict(
    num_envs=8192,
    batch_size=1024,
    num_minibatches=16,
    gait_freq_override=[0.5, 1.5],
)

# ---- Deploy defaults ----
DEPLOY_DEFAULTS = dict(
    command=[0.5, 0.2, 0.0],
    gait_freq=1.0,
    policy_path="policies/anymal",
    pert_enable=False,
    stairs_enabled=True,
    ros_velocity_scale=[0.5, 0.5, 0.5],
)

# ---- Environment configs ----
_global_pert_type = 0
_global_velocity_kick = [0.5, 2]
_global_kick_durations = [0.3, 1.8]
_global_kick_wait_times = [0.2, 2.0]

IMPL = "warp"
NCONMAX = 128 * TRAINING_DEFAULTS["num_envs"]
NJMAX = 512

KP=100
KD=1
def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=1000,
        vel_percentage=0.65,
        Kp=KP,
        Kd=KD,
        action_repeat=1,
        action_scale=0.5,
        history_len=2,
        history_update_steps=5,
        soft_joint_pos_limit_factor=0.95,
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
                linvel=0.1,
                heightscan=0.01,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                tracking_lin_vel=1.0,
                tracking_ang_vel=0.5,
                lin_vel_z=-2.0,
                ang_vel_xy=-0.05,
                orientation=-0.,
                dof_pos_limits=-0.0,
                pose=-1.,
                termination=-1.0,
                stand_still=-0.0,
                torques=-0.0002,
                action_rate=-0.01,
                energy=-0.001,
                feet_clearance=-0.,
                feet_height=-0.,
                feet_slip=-0.,
                feet_air_time=0.,
                feet_phase=1.0,
                feet_swing=0.,
                body_height=-0.,
                contact=0.2,
                center=-0.,
            ),
            tracking_sigma=0.25,
            swing_height=-0.35,
            base_feet_distance=-0.5,
            phase_sigma=0.04,
        ),
        pert_config=config_dict.create(
            enable=False,
            velocity_kick=_global_velocity_kick,
            kick_durations=_global_kick_durations,
            kick_wait_times=_global_kick_wait_times,
            type=_global_pert_type,
        ),
        command_config=config_dict.create(
            u_max=[1.5, 0.8, 1.2],
            u_min=[-1.5, -0.8, -1.2],
            b=[0.9, 0.25, 0.5],
        ),
        gait_freq=[0.5, 1.5],
        heighmap_size=(num_heightscans, num_widthscans),
    )


def baseline_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        vel_percentage=0.65,
        episode_length=1000,
        Kp=KP,
        Kd=KD,
        action_repeat=1,
        action_scale=0.5,
        history_len=2,
        history_update_steps=5,
        soft_joint_pos_limit_factor=0.95,
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
                linvel=0.1,
                heightscan=0.01,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                tracking_lin_vel=1.0,
                tracking_ang_vel=0.5,
                lin_vel_z=-2.0,
                ang_vel_xy=-0.05,
                orientation=-0.,
                dof_pos_limits=-0.,
                pose=-0.5,
                termination=-1.0,
                stand_still=-0.0,
                torques=-0.000001,
                action_rate=-0.001,
                energy=-0.0,
                feet_clearance=-0.0,
                feet_height=-0.,
                feet_slip=-0.0,
                feet_air_time=1.0,
                feet_phase=0.0,
                feet_swing=0.,
                body_height=-0.,
                contact=0.,
                center=-0.,
            ),
            tracking_sigma=0.25,
            swing_height=-0.35,
            base_feet_distance=-0.5,
            phase_sigma=0.05,
        ),
        pert_config=config_dict.create(
            enable=False,
            velocity_kick=_global_velocity_kick,
            kick_durations=_global_kick_durations,
            kick_wait_times=_global_kick_wait_times,
            type=_global_pert_type,
        ),
        command_config=config_dict.create(
            u_max=[1.5, 0.8, 1.2],
            u_min=[-1.5, -0.8, -1.2],
            b=[0.9, 0.25, 0.5],
        ),
        gait_freq=[0.5, 1.5],
        heighmap_size=(num_heightscans, num_widthscans),
    )


def wild_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        vel_percentage=0.65,
        episode_length=1000,
        Kp=KP,
        Kd=KD,
        action_repeat=1,
        action_scale=0.5,
        history_len=2,
        history_update_steps=5,
        soft_joint_pos_limit_factor=0.95,
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
                linvel=0.1,
                heightscan=0.01,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                tracking_lin_vel=1.0,
                tracking_ang_vel=0.5,
                lin_vel_z=-2.0,
                ang_vel_xy=-0.05,
                orientation=-0.2,
                dof_pos_limits=-1.0,
                pose=-0.2,
                termination=-1.0,
                stand_still=-0.,
                torques=-0.00002,
                action_rate=-0.005,
                energy=-0.0005,
                feet_clearance=0.1,
                feet_height=-0.,
                feet_slip=-0.1,
                feet_air_time=0.0,
                feet_phase=0.0,
                feet_swing=0.,
                body_height=-0.,
                contact=0.,
                center=-0.,
            ),
            tracking_sigma=0.25,
            swing_height=-0.35,
            base_feet_distance=-0.5,
            phase_sigma=0.05,
        ),
        pert_config=config_dict.create(
            enable=False,
            velocity_kick=_global_velocity_kick,
            kick_durations=_global_kick_durations,
            kick_wait_times=_global_kick_wait_times,
            type=_global_pert_type,
        ),
        command_config=config_dict.create(
            u_max=[1.5, 0.8, 1.2],
            u_min=[-1.5, -0.8, -1.2],
            b=[0.9, 0.25, 0.5],
        ),
        gait_freq=[0.5, 1.5],
        heighmap_size=(num_heightscans, num_widthscans),
    )
