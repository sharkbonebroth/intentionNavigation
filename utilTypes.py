import numpy as np
from typing import TypeAlias

Trajectory: TypeAlias = tuple[float, float]

class Action:
    def __init__(self, lin_x : float, ang_z : float):
        """
        TODO: is our action only lin x and ang z?
        """
        self.lin_x = lin_x
        self.ang_z = ang_z
        
    def get_as_ndarray(self) -> np.ndarray:
        return np.array([self.lin_x, self.ang_z])