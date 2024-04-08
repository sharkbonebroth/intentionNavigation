from __future__ import annotations
import numpy as np
import random
from pathlib import Path

MAPSCALE = 0.05 # each pixel is 0.05m

class Map:
  def __init__(self, mapGrid: np.ndarray):
    self.height = mapGrid.shape[0]
    self.width = mapGrid.shape[1]
    self.mapGrid = mapGrid
    self.name = ""

  @classmethod
  def fromFile(cls, filePath: str) -> Map:
    mapGrid = np.load(filePath, allow_pickle=True)
    mapName = Path(filePath).stem

    newMap = Map(mapGrid)
    newMap.setName(mapName)
          
    return newMap

  def setName(self, name: str):
    self.name = name

  def saveToFile(self, filePath: str):
    np.save(filePath, self.mapGrid)

  def getLidarMeasurement(self, x: float, y: float, rad: float) -> np.ndarray:
    pass

  def getColorImageNpArray(self):
    colorImageNpArray = np.zeros((self.height, self.width, 3), "uint8")
    for i in range(self.height):
      for j in range(self.width):
        if not self.mapGrid[i][j]:
          colorImageNpArray[i][j][0] = 255
          colorImageNpArray[i][j][1] = 255
          colorImageNpArray[i][j][2] = 255

    return colorImageNpArray

class mapGenerator:
  def __init__(self):
    pass

  def generatePerlinNoiseGrid(self, width: int, height: int, initialAliveProbability: float) -> np.ndarray:
    pass

  def generateRandomNoiseGrid(self, width: int, height: int, initialAliveProbability: float) -> np.ndarray:
    mapGrid = np.ones((height, width), dtype = bool)
    for i in range(height):
      for j in range(width):
        if random.random() < initialAliveProbability:
          mapGrid[i][j] = False

    return mapGrid

  def getNumSurroundingWalls(self, mapGrid: np.ndarray, x: int, y: int) -> int:
    width = mapGrid.shape[1]
    height = mapGrid.shape[0]

    numSurrounding = 0
    for i in [-1, 0, 1]:
      for j in [-1, 0, 1]:
        if (i == 0) and (j == 0):
          continue
        xBeingSampled = j + x
        yBeingSampled = i + y
        if (yBeingSampled < 0) or (yBeingSampled >= height) or (xBeingSampled < 0) or (xBeingSampled >= width):
          numSurrounding += 1
        elif mapGrid[yBeingSampled][xBeingSampled]:
          numSurrounding += 1

    return numSurrounding

  def generateMapCellularAutomata(
    self, 
    mapWidth: int, 
    mapHeight: int, 
    scaleFactor: int,
    noiseType: str,
    initialAliveProbability: float,
    birthThreshold: int,
    deathThreshold: int,
    numIters: int,
  ) -> Map:

    mapGridPrev = None
    mapGridNext = None

    # Generate the map on a smaller scale
    width = int(mapWidth / scaleFactor)
    height = int(mapHeight / scaleFactor)
    if (noiseType == "perlin"):
      mapGridNext = self.generatePerlinNoiseGrid(width, height, initialAliveProbability)
    elif (noiseType == "random"):
      mapGridNext = self.generateRandomNoiseGrid(width, height, initialAliveProbability)
    else:
      print("Unable to initialize map! Unknown noise type!") 
      return

    for iterNum in range(numIters):
      mapGridPrev = mapGridNext
      mapGridNext = np.ones((height, width), dtype = bool)
      for y in range(height):
        for x in range(width):
          numSurrounding = self.getNumSurroundingWalls(mapGridPrev, x, y)
          if mapGridPrev[y][x] and numSurrounding >= deathThreshold:
            mapGridNext[y][x] = False
          elif (not mapGridPrev[y][x]) and numSurrounding >= birthThreshold:
            mapGridNext[y][x] = False

    # Upscale the generated map grid
    scaledMapGrid = mapGridNext.repeat(scaleFactor, axis=0).repeat(scaleFactor, axis=1)
    return Map(scaledMapGrid)