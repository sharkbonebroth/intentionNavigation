import gymnasium as gym
from gymnasium import spaces
import os
import numpy as np
from typing import Tuple, List

from generateLabelledData import TrainingDataPoint
from map import Map
from path import AStarTrajectorySolver


class RobotNavigationEnvironment(gym.Env):
  metadata = {
    "render_modes": ["human", "rgb_array"], 
    "render_fps": 4
  }

  def getNearestPointAlongTrajectoryIndexAndDist(self, point: Tuple[int, int], trajectory: List[Tuple[int, int]]) -> Tuple[int, float]:
    closestDistSq = 10000000000
    closestIndex = 0
    for i in range(len(trajectory)):
      distSq = (trajectory[i][0] - point[0])**2 + (trajectory[i][1] - point[1])**2
      if distSq < closestDistSq:
        closestDistSq = distSq
        closestIndex = i

    return (closestIndex, closestDistSq)


  def __init__(self, trainingDataPoint: TrainingDataPoint, mapFolder: str, render_mode=None):
    # Load map
    mapPath = os.path.join(mapFolder, trainingDataPoint.mapName)
    map = Map.fromFile(mapPath)

    # Get trajectory
    trajectorySolver = AStarTrajectorySolver()
    trajectory = trajectorySolver.solveTrajectory(
      map, 
      trainingDataPoint.startPoint,
      trainingDataPoint.endPoint
    )

    # Use trajectory to generate reward data structure. Rewards will beased on 2 parts
    # 1: progress along the path
    # 2: Distance from walls

    numRows = map.mapGrid.shape[1]
    numCols = map.mapGrid.shape[0]
    self.nearestTrajectoryPointAndDistanceMat = np.empty(shape = (numRows, numCols, 2),dtype = float)
    # we dont actly need to calculate this for the whole image, but lets j do this first
    for row in range(numRows):
      for col in range(numCols):
        if map.mapGrid[row][col]:
          continue
        index, distance = self.getNearestPointAlongTrajectoryIndexAndDist((col, row), trajectory)
        self.nearestTrajectoryPointAndDistanceMat[row][col][0] = index
        self.nearestTrajectoryPointAndDistanceMat[row][col][1] = distance


    

