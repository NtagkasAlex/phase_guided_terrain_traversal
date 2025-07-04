import xml.etree.ElementTree as xml_et
import numpy as np
import cv2
import noise

ROBOT = "go2"
INPUT_SCENE_PATH = "./go2/xmls/scene_mjx_feetonly.xml"
OUTPUT_SCENE_PATH ="./go2/xmls/terrain_scene_mjx.xml"
TEST_SCENE_PATH ="./go2/xmls/terrain_test_mjx.xml"
DATA_SCENE_PATH="./go2/xmls/data.xml"
HUGE_STAIRS="./go2/xmls/huge_stairs.xml"

# INPUT_SCENE_PATH = "./terrain/scene.xml"
# OUTPUT_SCENE_PATH ="./terrain/scene_terrain.xml"
# TEST_SCENE_PATH ="./terrain/scene_test.xml"

# zyx euler angle to quaternion
def euler_to_quat(roll, pitch, yaw):
    cx = np.cos(roll / 2)
    sx = np.sin(roll / 2)
    cy = np.cos(pitch / 2)
    sy = np.sin(pitch / 2)
    cz = np.cos(yaw / 2)
    sz = np.sin(yaw / 2)

    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float64,
    )


# zyx euler angle to rotation matrix
def euler_to_rot(roll, pitch, yaw):
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ],
        dtype=np.float64,
    )

    rot_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rot_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


# 2d rotate
def rot2d(x, y, yaw):
    nx = x * np.cos(yaw) - y * np.sin(yaw)
    ny = x * np.sin(yaw) + y * np.cos(yaw)
    return nx, ny


# 3d rotate
def rot3d(pos, euler):
    R = euler_to_rot(euler[0], euler[1], euler[2])
    return R @ pos


def list_to_str(vec):
    return " ".join(str(s) for s in vec)

import random
import string

def random_box_name():
    suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=5))  # Generate a random 5-character string
    return f"box_{suffix}"

class TerrainGenerator:

    def __init__(self, width=None, step_height=None, num_stairs=None, length=None,render=False,max_bodies=None) -> None:
        self.scene = xml_et.parse(INPUT_SCENE_PATH)
        self.root = self.scene.getroot()
        self.worldbody = self.root.find("worldbody")
        self.asset = self.root.find("asset")
        self.count_boxes=0
        self.render=render
        self.max_bodies=max_bodies
        args = [width, step_height, num_stairs, length]
        if args.count(None) != 1:
            raise ValueError("Exactly three arguments must be provided.")
        if num_stairs is None:
            self.num_stairs = length / width  # Replace with actual formula
            self.width, self.step_height, self.length = width, step_height, length
        elif length is None:
            self.length = num_stairs * width  # Replace with actual formula
            self.width, self.step_height, self.num_stairs = width, step_height, num_stairs
        self.block_height=self.num_stairs*self.step_height

        self.box_data = [] 


    # Add Box to scene
    def AddBox(self,
               position=[1.0, 0.0, 0.0],
               euler=[0.0, 0.0, 0.0], 
               size=[2,2,2]):
        # position[2]-=0.02
        if self.render:
            if self.max_bodies is not None and self.count_boxes>=self.max_bodies-1:
                return
            # Create a new body inside worldbody
            box_body = xml_et.SubElement(self.worldbody, "body")
            box_body.attrib["pos"] = list_to_str(position)  # Set body position
            box_body.attrib["quat"] = list_to_str(euler_to_quat(euler[0], euler[1], euler[2]))
            box_body.attrib["name"] = random_box_name()


            geo = xml_et.SubElement(box_body, "geom")
            geo.attrib["type"] = "box"
            # geo.attrib["pos"] = list_to_str(position)  # Set body position
            # geo.attrib["quat"] = list_to_str(euler_to_quat(euler[0], euler[1], euler[2]))
            geo.attrib["size"] = list_to_str(0.5*np.array(size))  # Half size for Mujoco
            geo.attrib["contype"] = "2"
            geo.attrib["conaffinity"] = "1"
            self.count_boxes+=1


        else:
            quat = euler_to_quat(euler[0], euler[1], euler[2])
            self.box_data.append({"pos": np.array(position), "size": 0.5* np.array(size),"quat":np.array(quat)})
            self.count_boxes+=1


    
    def AddGeometry(self,
               position=[1.0, 0.0, 0.0],
               euler=[0.0, 0.0, 0.0], 
               size=[0.1, 0.1],geo_type="box"):
        
        # geo_type supports "plane", "sphere", "capsule", "ellipsoid", "cylinder", "box"
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = geo_type
        geo.attrib["size"] = list_to_str(
            0.5 * np.array(size))  # half size of box for mujoco
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    def AddStairs(self,
                  init_pos=[0.0, 0.0, 0.0],
                  yaw=0.0):
        length=self.length
        height=self.step_height
        stair_nums=self.num_stairs
        width=self.width

        local_pos = [ -width/2,length/2, 0.]
        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height
        
            x, y = rot2d(local_pos[0]-stair_nums*width/2, local_pos[1]-stair_nums*width/2, yaw)
            # x-=stair_nums*width/2
            # y-=stair_nums*width/2
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2]/2+init_pos[2]],
                        [0.0, 0.0, yaw], [width, length, local_pos[2]])
    def AddFlat(self,init_pos=[0.0, 0.0, 0.0],
                height=1.,
                width =0.1,):
        length=self.length
        if height>0.:
            self.AddBox([init_pos[0],init_pos[1], height/2],
                        [0.0, 0.0,0.], [length,length,height])
        else:
            self.AddBox([init_pos[0],init_pos[1], height-width/2],
                        [0.0, 0.0,0.], [length,length,width])
    
    def AddTurningStairsUp(self, init_pos=[0.0, 0.0, 0.0],
                        yaw=np.pi/2):
        width=self.width
        height=self.step_height
        stair_nums=self.num_stairs 
        local_pos = [ -stair_nums*width/2-width/2,+stair_nums*width/2, 0.]
        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[1] -= width/2
            local_pos[2] += height
        
            x, y = rot2d(local_pos[0],local_pos[1], yaw)
            # x-=stair_nums*width/2
            # y-=stair_nums*width/2
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2]/2+init_pos[2]],
                        [0.0, 0.0, yaw], [width, width + width * i, local_pos[2]])
        local_pos = [stair_nums*width/2-width/2,stair_nums*width/2, height]
        for i in range(stair_nums-1):
            local_pos[0] -= width
            local_pos[1] -= width/2
            local_pos[2] += height
        
            x, y = rot2d(local_pos[0], local_pos[1] , yaw+np.pi/2)
            # x-=stair_nums*width/2
            # y-=stair_nums*width/2
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2]/2+init_pos[2]],
                        [0.0, 0.0, yaw+np.pi/2], [width, width + width * i,  local_pos[2]])
    
            

    def AddTurningStairsDown(self, init_pos=[0.0, 0.0, 0.0],
                        yaw=np.pi/2):
        width=self.width
        height=self.step_height
        stair_nums=self.num_stairs 
        local_pos = [ -stair_nums*width/2-width/2,+stair_nums*width/2, stair_nums*height+height]
        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[1] -= width/2
            local_pos[2] -= height
        
            x, y = rot2d(local_pos[0],local_pos[1], yaw)
            # x-=stair_nums*width/2
            # y-=stair_nums*width/2
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2]/2+init_pos[2]],
                        [0.0, 0.0, yaw], [width, width + width * i, local_pos[2]])
        local_pos = [stair_nums*width/2-width/2,stair_nums*width/2, (stair_nums-1)*height+height]
        for i in range(stair_nums-1):
            local_pos[0] -= width
            local_pos[1] -= width/2
            local_pos[2] -= height
        
            x, y = rot2d(local_pos[0], local_pos[1] , yaw+np.pi/2)
            # x-=stair_nums*width/2
            # y-=stair_nums*width/2
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2]/2+init_pos[2]],
                        [0.0, 0.0, yaw+np.pi/2], [width, width + width * i,  local_pos[2]])
        
        
    
    def AddRoughGround(self,
                       init_pos=[1.0, 0.0, 0.0],
                       euler=[0.0, -0.0, 0.0],
                       nums=[10, 10],
                       box_size=[0.5, 0.5, 0.5],
                       box_euler=[0.0, 0.0, 0.0],
                       separation=[0.2, 0.2],
                       box_size_rand=[0.05, 0.05, 0.05],
                       box_euler_rand=[0.2, 0.2, 0.2],
                       separation_rand=[0.05, 0.05]):

        local_pos = [0.0, 0.0, -0.5 * box_size[2]]
        new_separation = np.array(separation) + np.array(
            separation_rand) * np.random.uniform(-1.0, 1.0, 2)
        for i in range(nums[0]):
            local_pos[0] += new_separation[0]
            local_pos[1] = 0.0
            for j in range(nums[1]):
                new_box_size = np.array(box_size) + np.array(
                    box_size_rand) * np.random.uniform(-1.0, 1.0, 3)
                new_box_euler = np.array(box_euler) + np.array(
                    box_euler_rand) * np.random.uniform(-1.0, 1.0, 3)
                new_separation = np.array(separation) + np.array(
                    separation_rand) * np.random.uniform(-1.0, 1.0, 2)

                local_pos[1] += new_separation[1]
                pos = rot3d(local_pos, euler) + np.array(init_pos)
                self.AddBox(pos, new_box_euler, new_box_size)

    def Save(self,filename=None):
        if filename is None:
            self.scene.write(OUTPUT_SCENE_PATH)
        else:
            self.scene.write(filename)
        
from wfc.wfc.wfc import WFCCore
from getIndexes import *

def generate_14(size):
    left=(-1,0)
    right=(1,0)
    up=(0,1)
    down=(0,-1)

    directions=[left,down,right,up]
    connections = {
        0: {left: (0, 4,10,11),    down: (0,5,11,12),    right: (0, 2,12,13),    up: (0, 3,13,10)},  
        1: {left: (1, 2,6,7), down: (1, 3,7,8), right: (1, 4,8,9), up: (1,5,9,6)}
    }
    
    connections.update(Stairs(directions))
    connections.update(StairsTurningUp(directions))
    connections.update(StairsTurningDown(directions))
    

    wfc = WFCCore(14, connections, (size,size))
# 
    # wfc.init_randomly()
    # Top and bottom rows
    outer=1
    for x in range(size):
        wfc.init((x, 0), outer)          # Top row
        wfc.init((x, size - 1), outer)   # Bottom row

    # Left and right columns (excluding corners to avoid duplication)
    for y in range(1, size - 1):
        wfc.init((0, y), outer)          # Left column
        wfc.init((size - 1, y), outer)   # Right column
    if np.random.random()>0.0:
        
        wfc.init((size//2,size//2),0)
    else:
        wfc.init((size//2,size//2),1)
    wfc.solve()
    wave = wfc.wave.wave
    
    return wave

def addElement(map:TerrainGenerator,index,pos):
    height=map.block_height
    if index==0:
        return
        map.AddFlat(init_pos=[pos[0],pos[1],0.],height=0.)
    elif index==1:
        map.AddFlat(init_pos=[pos[0],pos[1],0.],height=height)
    elif index==2:
        map.AddStairs(init_pos=[pos[0],pos[1],0],yaw=0)
    elif index==3:
        map.AddStairs(init_pos=[pos[0],pos[1],0],yaw=np.pi/2)
    elif index==4:
        map.AddStairs(init_pos=[pos[0],pos[1],0],yaw=np.pi)
    elif index==5:
        map.AddStairs(init_pos=[pos[0],pos[1],0],yaw=-np.pi/2)
    elif index==6:
        map.AddTurningStairsUp(init_pos=[pos[0],pos[1],0.],yaw=0.)
    elif index==7:
        map.AddTurningStairsUp(init_pos=[pos[0],pos[1],0.],yaw=np.pi/2)
    elif index==8:
        map.AddTurningStairsUp(init_pos=[pos[0],pos[1],0.],yaw=np.pi)
    elif index==9:
        map.AddTurningStairsUp(init_pos=[pos[0],pos[1],0.],yaw=-np.pi/2)
    elif index==10:
        map.AddTurningStairsDown(init_pos=[pos[0],pos[1],0],yaw=0)
    elif index==11:
        map.AddTurningStairsDown(init_pos=[pos[0],pos[1],0],yaw=np.pi/2)
    elif index==12:
        map.AddTurningStairsDown(init_pos=[pos[0],pos[1],0],yaw=np.pi)
    elif index==13:
        map.AddTurningStairsDown(init_pos=[pos[0],pos[1],0],yaw=-np.pi/2)

def create_centered_grid(N, d):
    half_size = (N - 1) / 2  
    x = (np.arange(N) - half_size) * d
    y = (np.arange(N) - half_size) * d
    X, Y = np.meshgrid(x, y, indexing='ij')  
    grid = np.stack((X, Y), axis=-1) 
    return grid
import jax.numpy as jnp
def create_random_matrix(num_envs,num_bodies,size,height_min,height_max):

    result_matrix = jnp.ones((num_envs, num_bodies, 10))
    incremented_values = jnp.arange(100, 100 + num_envs * num_bodies).reshape(num_envs, num_bodies, 1)
    result_matrix = result_matrix.at[..., :3].set(incremented_values)    
    result_matrix = result_matrix.at[..., 3:7].set(jnp.array([1,0,0,0]))
    for env_id in range(num_envs):
        height = np.random.uniform(height_min,height_max)
        width = np.random.uniform(0.3, 0.45)
        num_steps = np.random.choice([ 2,3,4])

        tg = TerrainGenerator(width=width, step_height=height, num_stairs=num_steps, render=False)
        wave = generate_14(size=size)
        grid = create_centered_grid(size, tg.length)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                addElement(tg, wave[i, j], grid[i, j])
        print(tg.count_boxes)

        for i,box in enumerate(tg.box_data):
            result_matrix = result_matrix.at[env_id,i,:3].set(box["pos"])
            result_matrix = result_matrix.at[env_id,i,3:7].set(box["quat"])
            result_matrix = result_matrix.at[env_id,i,7:].set(box["size"])
    return result_matrix

def random_test_env(num_bodies,size):


    # height = np.random.uniform(0.05, 0.15)
    # width = np.random.uniform(0.15, 0.35)
    # num_steps = np.random.choice([3, 4, 5, 6])
    num_steps=3
    width=0.4
    step_height=0.1
    # print(height,width,num_steps)
    tg = TerrainGenerator(width=width,step_height=step_height,num_stairs=num_steps,render=True)
    wave=generate_14(size=size)
    grid = create_centered_grid(size, tg.length)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            addElement(tg,wave[i,j],grid[i,j])
            
    print(tg.count_boxes)
    if tg.count_boxes<num_bodies-1:
        for i in range(num_bodies-tg.count_boxes-1):
            tg.AddBox([100+i,100+i,10])

    tg.Save(TEST_SCENE_PATH)
    
if __name__ == "__main__":

    size=5
    length=None
    num_steps=3
    width=0.1
    step_height=0.15
    num_envs,num_objects=100,100

    np.set_printoptions(precision=2,suppress=True)


    # # Uncomment to create terrains for DR
    # numbers=["01","02","03","04","05","06","07","08","09"]
    # for number in numbers:
    #     value = float("0." + number)
    #     print(value)  # 0.04
    #     res=create_random_matrix(num_envs,num_objects,size,value,value)
    #     np.save(f"./terrains/level{number}",res)
    #     print(res.shape)

    # exit()

    #filling
    # tg = TerrainGenerator(width=width,step_height=step_height,num_stairs=num_steps,render=True)
    # wave=generate_14(size=size)
    # grid = create_centered_grid(size, tg.length)
    # for i in range(num_objects):
    #     tg.AddBox([100+i,100+i,10])
    # tg.Save()

    #Uncomment for one random terrain for testing.
    random_test_env(num_objects,size)
    exit()
