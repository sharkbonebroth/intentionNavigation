import numpy as np
import enum
from typing import Tuple, List

trajectoryType = List[Tuple[float, float]]

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

class Intention(enum.Enum):
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