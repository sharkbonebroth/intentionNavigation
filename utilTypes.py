import numpy as np
import enum
from typing import Tuple, List

trajectoryType = List[Tuple[float, float]]

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def get_angle(pt1, pt2):
    pt1 = np.array(pt1[:2])
    pt2 = np.array(pt2[:2])
    
    pt1 = unit_vector(pt1)
    pt2 = unit_vector(pt2)
    
    return np.arccos(np.clip(np.dot(pt1, pt2), -1.0, 1.0))
    
def get_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1[:2]) - np.array(pt2[:2]))

def find_closest_point(point, traj : trajectoryType) -> int:
    closestId = 0
    closestDist = float('inf')
    for id in range(len(traj)):
        trajPoint = traj[id]
        dist = np.linalg.norm(np.array(point) - np.array(trajPoint))
        if dist < closestDist:
            closestDist = dist
            closestId =  id
    return closestId

def getIntentionAsOnehot(intentions, onehotSize):
    batch_size = intentions.size(dim=0)
    onehot = np.zeros((batch_size, onehotSize), dtype=np.float32)
    for i in range(batch_size):
        intention = intentions[i]
        if intention == -1.0:
            onehot[i,0] = 1
        elif intention == 0.0: 
            onehot[i,1] = 1
        elif intention == 1.0:
            onehot[i,2] = 1
    return onehot

class Intention(int, enum.Enum):
  LEFT = -1
  STRAIGHT = 0
  RIGHT = 1

class Action:
    def __init__(self, lin_x : float, ang_z : float):
        """
        TODO: is our action only lin x and ang z?
        """
        self.lin_x = lin_x
        self.ang_z = ang_z
        
    def get_as_ndarray(self) -> np.ndarray:
        return np.array([self.lin_x, self.ang_z])