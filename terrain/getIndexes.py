import numpy as np

def _indexing(yaw,indexes):
    yaw=np.mod(yaw,2*np.pi)
    yaw_map = {
        0.0: indexes[0],
        np.pi / 2: indexes[1],
        np.pi: indexes[2],
        3 * np.pi / 2: indexes[3]
    }

    for key, index in yaw_map.items():
        if np.isclose(yaw, key):
            return index
def S_index(yaw):
    indexes=[2,3,4,5]
    return _indexing(yaw,indexes)

def Su_index(yaw):
    indexes=[6,7,8,9]
    return _indexing(yaw,indexes)

def Sd_index(yaw):
    indexes=[10,11,12,13]
    return _indexing(yaw,indexes)


def StairsTurningUp(directions) -> dict:
    yaws = [0, np.pi/2, np.pi, 3*np.pi/2]
    all_connections = {}

    for i, yaw in enumerate(yaws):
        all_connections[Su_index(yaw)] = {
            directions[(0 + i) % 4]: (Su_index(yaw - np.pi/2), S_index(yaw - np.pi/2), Sd_index(yaw + np.pi)),
            directions[(1 + i) % 4]: (S_index(yaw + np.pi/2), 1),
            directions[(2 + i) % 4]: (S_index(yaw + np.pi), 1),
            directions[(3 + i) % 4]: (Su_index(yaw + np.pi/2), S_index(yaw), Sd_index(yaw + np.pi)),
        }
    
    return all_connections
def Stairs_simple(directions) -> dict:
    yaws = [0, np.pi/2, np.pi, 3*np.pi/2]
    all_connections = {}

    for i, yaw in enumerate(yaws):
        all_connections[S_index(yaw)] = {
            directions[(0 + i) % 4]: (S_index(yaw+np.pi),0),
            directions[(1 + i) % 4]: (0,S_index(yaw),S_index(yaw+np.pi/2),S_index(yaw-np.pi/2)),
            directions[(2 + i) % 4]: (S_index(yaw + np.pi), 1),
            directions[(3 + i) % 4]: (0,S_index(yaw),S_index(yaw+np.pi/2),S_index(yaw-np.pi/2)),
        }
    
    return all_connections
def Stairs(directions) -> dict:
    yaws = [0, np.pi/2, np.pi, 3*np.pi/2]
    all_connections = {}

    for i, yaw in enumerate(yaws):
        all_connections[S_index(yaw)] = {
            directions[(0 + i) % 4]: (S_index(yaw+np.pi),0),
            directions[(1 + i) % 4]: (S_index(yaw),Su_index(yaw),Sd_index(yaw-np.pi/2)),
            directions[(2 + i) % 4]: (S_index(yaw + np.pi), 1),
            directions[(3 + i) % 4]: (Su_index(yaw + np.pi/2), S_index(yaw), Sd_index(yaw + np.pi)),
        }
    
    return all_connections
def StairsTurningDown(directions) -> dict:
    yaws = [0, np.pi/2, np.pi, 3*np.pi/2]
    all_connections = {}

    for i, yaw in enumerate(yaws):
        all_connections[Sd_index(yaw)] = {
            directions[(0 + i) % 4]: (S_index(yaw+np.pi/2),Su_index(yaw+np.pi),Sd_index(yaw-np.pi/2)),
            directions[(1 + i) % 4]: (0,S_index(yaw-np.pi/2)),
            directions[(2 + i) % 4]: (S_index(yaw), 0),
            directions[(3 + i) % 4]: (Su_index(yaw + np.pi), S_index(yaw+np.pi), Sd_index(yaw + np.pi/2)),
        }
    
    return all_connections
