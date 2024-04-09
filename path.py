from __future__ import annotations
from typing import Tuple, List
from map import Map
from scipy import ndimage
import heapq
import numpy as np
import math
from skimage.draw import line

class Trajectory:
  def __init__(self, startPoint: Tuple[int, int], endPoint: Tuple[int, int], map: Map):
    self.pathWaypoints = []
    pass

class AStarNode:
  def __init__(self, coords: Tuple[int, int], parent: AStarNode, g: float):
    self.coords = coords # x, y
    self.parent = parent
    self.g = g

class AStarTrajectorySolver:
  def __init__(self):
    pass

  def expandNode(self, node: AStarNode, mapGrid: np.ndarray) -> List[AStarNode]:
    childrenNodes = []

    for i in [-1, 0, 1]:
      for j in [-1, 0, 1]:
        if i == 0 and j == 0:
          continue
        col = node.coords[0] + i
        row = node.coords[1] + j
        g = (i**2 + j**2)**0.5 + node.g
        if (0 <= row < mapGrid.shape[0] and 0 <= col < mapGrid.shape[1]):
          if not mapGrid[row][col]:
            childrenNodes.append(AStarNode((col, row), node, g))
          
    return childrenNodes

  def getEuclideanDistance(self, start: Tuple[int, int], end: Tuple[int, int]) -> float:
    return ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5

  def getManhattanDistance(self, start: Tuple[int, int], end: Tuple[int, int]) -> float:
    return abs(end[0] - start[0]) + abs(end[1] - start[1])

  def inLineOfSight(self, coordA: Tuple[int, int], coordB: Tuple[int, int], map: Map) -> bool:
    # Damn lazy to do this properly 
    rr, cc = line(coordA[1], coordA[0], coordB[1], coordB[0])
    for i in range(len(rr)):
      if map.mapGrid[rr[i]][cc[i]]:
        return False
    # Should we encode some vehicle dynamics here?
    return True

  def solveTrajectory(self, map: Map, startPoint: Tuple[int, int, float], endPoint: Tuple[int, int, float]) -> Trajectory:
    mapGrid = map.mapGrid

    # Perform dilation operation for inflation zones
    structuringElement = ndimage.generate_binary_structure(2, 1)
    structuringElement = ndimage.iterate_structure(structuringElement, 20).astype(bool)
    mapGridInflated = ndimage.binary_dilation(mapGrid, structure=structuringElement).astype(mapGrid.dtype)

    # Downsample map
    downsampleRate = 4 # Assuming a map scale of 1 px to 5cm, this downsamples it to 1 px per 20cm
    mapGridDownsampled = mapGridInflated[::downsampleRate, ::downsampleRate]
    startPointDownSampled = (int(startPoint[0] / downsampleRate), int(startPoint[1] / downsampleRate))
    endPointDownSampled = (int(endPoint[0] / downsampleRate), int(endPoint[1] / downsampleRate))
 
    # Solve A*
    path = [] # list of tuple[int, int]
    frontier = [(self.getEuclideanDistance(startPointDownSampled, endPointDownSampled), 0, AStarNode(startPointDownSampled, None, 0))]
    minCostMat = np.ones(mapGridDownsampled.shape)
    minCostMat *= np.inf
    minCostMat[startPointDownSampled[1]][startPointDownSampled[0]] = 0
    visitedMat = np.zeros(mapGridDownsampled.shape, dtype = bool)
    heapq.heapify(frontier)
    seqNo = 0
    found = False
    while len(frontier) > 0 and not found:
      nextNode = heapq.heappop(frontier)[2]
      visitedMat[nextNode.coords[1]][nextNode.coords[0]] = True
      for childNode in self.expandNode(nextNode, mapGridDownsampled):
        if childNode.coords == endPointDownSampled:
          found = True
          backtrackingHead = childNode
          path.append(backtrackingHead.coords)
          while (backtrackingHead.parent != None):
            path.append(backtrackingHead.parent.coords)
            backtrackingHead = backtrackingHead.parent
          path = path[::-1]
          break
        else:
          if visitedMat[childNode.coords[1]][childNode.coords[0]] == False:
            if minCostMat[childNode.coords[1]][childNode.coords[0]] > childNode.g:
              minCostMat[childNode.coords[1]][childNode.coords[0]] = childNode.g
              value = self.getEuclideanDistance(childNode.coords, endPointDownSampled) + childNode.g # h + g
              seqNo += 1
              heapq.heappush(frontier, (value, seqNo, childNode))
      
    if path == []:
      print("Error! Unable to find path")
    # return [(coord[0] * downsampleRate, coord[1] * downsampleRate) for coord in path]

    # Perform smoothing on original scale. 
    smoothedPath = [(startPoint[0], startPoint[1])]
    for i in range(1, len(path) - 1):
      pathCoordOriginalScale = (path[i][0] * downsampleRate, path[i][1] * downsampleRate)
      if not self.inLineOfSight(smoothedPath[-1], pathCoordOriginalScale, map) or i%10 == 0:
        smoothedPath.append(pathCoordOriginalScale)
        
    smoothedPath.append((endPoint[0], endPoint[1]))
    # Interpolation using centripetal catmull rom spline. 
    # Using this has the neat effect of weighing initial orientation too
    alpha = 0.5 # alpha parameter. 0.5 is centripetal

    initialControlPtx = -math.cos(startPoint[2]) * 200 + startPoint[0]
    initialControlPty = -math.sin(startPoint[2]) * 200 + startPoint[1]
    finalControlPtx = math.cos(endPoint[2]) * 200 + endPoint[0]
    finalControlPty = math.sin(endPoint[2]) * 200 + endPoint[1]

    smoothedPath.insert(0, (initialControlPtx, initialControlPty))
    smoothedPath.append((finalControlPtx, finalControlPty))

    trajectory = []
    addedMat = np.zeros(mapGrid.shape, dtype=bool)
    tParameterized = [0]
    for i in range(1, len(smoothedPath)):
      dx = smoothedPath[i][0] - smoothedPath[i - 1][0]
      dy = smoothedPath[i][1] - smoothedPath[i - 1][1]
      l = (dx**2 + dy**2)**0.5
      dt = l**alpha
      tParameterized.append(tParameterized[-1] + dt)

    for i in range(1, len(smoothedPath) - 2):
      dx = smoothedPath[i + 1][0] - smoothedPath[i][0]
      dy = smoothedPath[i + 1][1] - smoothedPath[i][1]

      numIncrements = int((dx**2 + dy**2)**0.5 * 2)

      for t in np.linspace(tParameterized[i], tParameterized[i+1],numIncrements):
        t0 = tParameterized[i-1]
        t1 = tParameterized[i]
        t2 = tParameterized[i+1]
        t3 = tParameterized[i+2]

        P0x = smoothedPath[i-1][0]
        P1x = smoothedPath[i][0]
        P2x = smoothedPath[i+1][0]
        P3x = smoothedPath[i+2][0]

        P0y = smoothedPath[i-1][1]
        P1y = smoothedPath[i][1]
        P2y = smoothedPath[i+1][1]
        P3y = smoothedPath[i+2][1]

        L01x = (t1-t)/(t1-t0) * P0x + (t-t0)/(t1-t0) * P1x
        L12x = (t2-t)/(t2-t1) * P1x + (t-t1)/(t2-t1) * P2x
        L23x = (t3-t)/(t3-t2) * P2x + (t-t2)/(t3-t2) * P3x
        L012x = (t2-t)/(t2-t0) * L01x + (t-t0)/(t2-t0) * L12x
        L123x = (t3-t)/(t3-t1) * L12x + (t-t1)/(t3-t1) * L23x
        x = (t2-t)/(t2-t1) * L012x + (t-t1)/(t2-t1) * L123x

        L01y = (t1-t)/(t1-t0) * P0y + (t-t0)/(t1-t0) * P1y
        L12y = (t2-t)/(t2-t1) * P1y + (t-t1)/(t2-t1) * P2y
        L23y = (t3-t)/(t3-t2) * P2y + (t-t2)/(t3-t2) * P3y
        L012y = (t2-t)/(t2-t0) * L01y + (t-t0)/(t2-t0) * L12y
        L123y = (t3-t)/(t3-t1) * L12y + (t-t1)/(t3-t1) * L23y
        y = (t2-t)/(t2-t1) * L012y + (t-t1)/(t2-t1) * L123y

        xi = int(x)
        yi = int(y)
        if not addedMat[yi][xi]:
          addedMat[yi][xi] = True
          trajectory.append((xi, yi))     

    return trajectory