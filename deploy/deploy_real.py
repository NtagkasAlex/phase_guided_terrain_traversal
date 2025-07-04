import time
import sys
import torch 
import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_

from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
from  unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
# import configs.unitree_legged_const as go2
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
import policy_net as pn
# from sensor_msgs_py import point_cloud2
# from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
from builtin_interfaces.msg import Time  # This is the proper type
from std_msgs.msg import Header
import copy
import array
from collections import namedtuple
import sys
from typing import Iterable, List, NamedTuple, Optional

import numpy as np
try:
    from numpy.lib.recfunctions import (structured_to_unstructured, unstructured_to_structured)
except ImportError:
    from sensor_msgs_py.numpy_compat import (structured_to_unstructured,
                                             unstructured_to_structured)

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from go2.go2_constants import *
filename="i_believe"
filename="policy_folder/policy177"

PHASES=np.array([0.,np.pi,np.pi,0.])
ctrl_dt=0.02
pd_dt=0.02
freq=2.
cmd_x=0.1
reorder=[3,4,5,0,1,2,9,10,11,6,7,8]
PosStopF = 2.146e9
VelStopF = 16000.0
cmd_scale=np.array([0.5,0.5,0.8])

def read_points(
        cloud: PointCloud2,
        field_names: Optional[List[str]] = None,
        skip_nans: bool = False,
        uvs: Optional[Iterable] = None,
        reshape_organized_cloud: bool = False) -> np.ndarray:
    """
    Wrapper for ros2 point cloud and unitree point cloud. 
    Copied directly from point cloud library.
    """

    required_attrs = ['width', 'height', 'fields', 'point_step', 'data', 'is_bigendian', 'is_dense']
    for attr in required_attrs:
        assert hasattr(cloud, attr), f"Missing attribute: {attr}"

    if isinstance(cloud.data, list):
        buffer = bytes(cloud.data)
    else:
        buffer = cloud.data

    points = np.ndarray(
        shape=(cloud.width * cloud.height, ),
        dtype=pc2.dtype_from_fields(cloud.fields, point_step=cloud.point_step),
        buffer=buffer)


    if field_names is not None:
        assert all(field_name in points.dtype.names for field_name in field_names), \
            'Requests field is not in the fields of the PointCloud!'

        points = points[list(field_names)]


    if bool(sys.byteorder != 'little') != bool(cloud.is_bigendian):
        points = points.byteswap(inplace=True)

 
    if skip_nans and not cloud.is_dense:
        
        not_nan_mask = np.ones(len(points), dtype=bool)
        for field_name in points.dtype.names:
        
            not_nan_mask = np.logical_and(
                not_nan_mask, ~np.isnan(points[field_name]))
        
        points = points[not_nan_mask]

    
    if uvs is not None:
    
        if not isinstance(uvs, np.ndarray):
            uvs = np.fromiter(uvs, int)
    
        points = points[uvs]

    
    if reshape_organized_cloud and cloud.height > 1:
        points = points.reshape(cloud.width, cloud.height)

    return points
import cv2
import numpy as np


class Custom():

    def __init__(self):
        self.Kp = 60.0 #60.0 #20 #40
        self.Kd = 3.0 #2.0   #0.5  #1 
        self.time_consume = 0
        self.rate_count = 0
        self.sin_count = 0
        self.motiontime = 0

        self.low_cmd = unitree_go_msg_dds__LowCmd_()  
        self.low_state = None  
        self._targetPos_1 = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
                             -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
        # self._targetPos_2 = [0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
        #                     0.0, 0.67, -1.3, 0.0, 0.67, -1.3]

        self._targetPos_2= [0 ,0.9 ,-1.8 ,0 ,0.9, -1.8, 0 ,0.9, -1.8, 0 ,0.9, -1.8]
        self._targetPos_3= self._targetPos_2
        self._targetPos_3=self._targetPos_1
        # self._targetPos_3 = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
        #                      -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]

        self.startPos = [0.0] * 12
        self.duration_1 = int(2/pd_dt)
        self.duration_2 = int(0.5/pd_dt)
        self.duration_3 = int(3/pd_dt)
        self.duration_4 = int(100/pd_dt)
        self.duration_5 = int(4/pd_dt)

        self.percent_1 = 0
        self.percent_2 = 0
        self.percent_3 = 0
        self.percent_4 = 0
        self.percent_5 = 0 
        self.firstRun = True
        self.done = False

        # thread handling
        self.lowCmdWriteThreadPtr = None

        self.crc = CRC()
        
        self.counter=0
        self.qj=np.zeros(12)
        self.dqj=np.zeros(12)
        self.cmd=np.zeros(3)
        # self.obs=np.zeros(120)
        self.remote_controller = RemoteController()
        self.action=np.zeros(12)
        self.default_pos=np.array(self._targetPos_2)
        self.action_scale=0.5
        self.policy_network=pn.policy_net(policy_file=filename)
        self.phase=PHASES
        self.dt=ctrl_dt
        # self.lin_vel=np.zeros(3)
        self.heightmap = np.full((num_heightscans, num_widthscans), 0.0)
        self.emergency=False
    def Init(self):
        self.InitLowCmd()

        # create publisher #
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)


        # self.lin_vel_subscriber = ChannelSubscriber("rt/aft_mapped_to_init", Odometry_)
        # self.lin_vel_subscriber.Init(self.LinVelMessageHandler, 10)


        self.heightmap_subscriber  = ChannelSubscriber("rt/elevation_heightmap", PointCloud2_)
        self.heightmap_subscriber.Init(self.HeightMapMessageHandler, 10)


        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result['name']:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=pd_dt, target=self.LowCmdWrite, name="writebasiccmd"
        )
        self.lowCmdWriteThreadPtr.Start()

    # Private methods
    def InitLowCmd(self):
        self.low_cmd.head[0]=0xFE
        self.low_cmd.head[1]=0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q= PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)
        for i in range(12):
            index=reorder[i]
            self.qj[index] = self.low_state.motor_state[i].q
            self.dqj[index] = self.low_state.motor_state[i].dq

    # def LinVelMessageHandler(self, msg: Odometry_):
    #     linear = msg.twist.twist.linear        
    #     self.lin_vel[0]=linear.x
    #     self.lin_vel[1]=linear.y
    #     self.lin_vel[2]=linear.z
        # print(linear)
        # print("FR_0 motor state: ", msg.motor_state[go2.LegID["FR_0"]])
        # print("IMU state: ", msg.imu_state)
        # print("Battery state: voltage: ", msg.power_v, "current: ", msg.power_a)
    def HeightMapMessageHandler(self,msg:PointCloud2_):
        
        width = msg.width
        height = msg.height

        for i, point in enumerate(read_points(msg, field_names=("x", "y", "z"), skip_nans=False)):
            _, _, z = point
            row = i // width
            col = i % width
            self.heightmap[row, col] = z
        
    def LowCmdWrite(self):
        if  self.remote_controller.button[KeyMap.A] == 1:
            self.emergency=True
        if self.firstRun:
            for i in range(12):
                self.startPos[i] = self.low_state.motor_state[i].q
            self.firstRun = False

        self.percent_1 += 1.0 / self.duration_1
        self.percent_1 = min(self.percent_1, 1)
        if self.percent_1 < 1:
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - self.percent_1) * self.startPos[i] + self.percent_1 * self._targetPos_1[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0

        if (self.percent_1 == 1) and (self.percent_2 <= 1):
            self.percent_2 += 1.0 / self.duration_2
            self.percent_2 = min(self.percent_2, 1)
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - self.percent_2) * self._targetPos_1[i] + self.percent_2 * self._targetPos_2[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0

        if (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 < 1):
            self.percent_3 += 1.0 / self.duration_3
            self.percent_3 = min(self.percent_3, 1)
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = self._targetPos_2[i] 
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0

        if (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 == 1) and (self.percent_4 < 1):
            self.percent_4 += 1.0 / self.duration_4
            self.percent_4 = min(self.percent_4, 1)
            policy_target=self.run_policy()
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = policy_target[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0
        if (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 == 1) and (self.percent_4 == 1) and (self.percent_5<=1) or self.emergency:

            if self.percent_5<=1e-10:
                for i in range(12):
                    self.startPos[i] = self.low_state.motor_state[i].q
            self.percent_5 += 1.0 / self.duration_5
            self.percent_5 = min(self.percent_5, 1)
            
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - self.percent_5) * self.startPos[i] + self.percent_5 * self._targetPos_3[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0
        
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)
    def run_policy(self):
        self.counter += 1
        
        quat = self.low_state.imu_state.quaternion
        
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32).reshape(3)
        
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        gait_freq=freq


        self.phase+=2*np.pi*gait_freq*self.dt
        self.phase=np.fmod(self.phase,2*np.pi)
        
        sin = np.sin(self.phase)
        cos = np.cos(self.phase)
        phase_obs = np.concatenate([cos, sin])
        z_values=self.heightmap.ravel()
        z_normal=z_values-np.min(z_values)
 
        self.cmd[0] = cmd_scale[0]*self.remote_controller.ly
        self.cmd[1] = cmd_scale[1]*self.remote_controller.lx * -1
        self.cmd[2] = cmd_scale[2]*self.remote_controller.rx * -1

        num_actions = 12

        obs = np.hstack([
                # linvel,#3
                ang_vel,#3
                gravity_orientation,#3
                qj_obs-self.default_pos,#12
                dqj_obs,#12
                phase_obs,#8
                z_normal,#N*M
                gait_freq,#1
                self.action,#12
                self.cmd,#3
        ])        # self.obs[:3]
        
        obs_tensor=torch.tensor(np.asarray(obs).copy(), dtype=torch.float32).reshape((1,-1))
        self.action = self.policy_network(obs_tensor).detach().numpy().squeeze()
        target_dof_pos = self.default_pos + self.action * self.action_scale
        return target_dof_pos
    
    
if __name__ == '__main__':
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()
    custom.Start()
    
    

    while True:        
        if custom.percent_5 == 1.0: 
            time.sleep(1)
            print("Done!")
            sys.exit(-1)     
        time.sleep(1)