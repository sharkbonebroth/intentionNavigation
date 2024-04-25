from __future__ import annotations
import numpy as np
import random
from pathlib import Path
from typing import Tuple
from scipy import ndimage
from skimage.draw import rectangle
from utilTypes import trajectoryType

MAPSCALE = 0.1 # each pixel is 0.1m

class Map:
  def __init__(self, mapGrid: np.ndarray):
    self.height = mapGrid.shape[0]
    self.width = mapGrid.shape[1]
    self.mapGrid = mapGrid
    self.inflationZoneGrid = self.getInflationZoneGrid(mapGrid)
    self.colorImageNpArray = self.getColorImageNpArray()
    self.colorImageNpArrayWithTrajPlotted = None
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

  def getInflationZoneGrid(self, mapGrid):
    structuringElement = ndimage.generate_binary_structure(2, 1)
    structuringElement = ndimage.iterate_structure(structuringElement, 10).astype(bool)
    return ndimage.binary_dilation(mapGrid, structure=structuringElement).astype(mapGrid.dtype)

  def getColorImageNpArray(self, scaleFactor: int = 1):
    colorImageNpArray = np.zeros((self.height, self.width, 3), "uint8")
    
    for i in range(self.height):
      for j in range(self.width):
        if not self.mapGrid[i][j]:
          if self.inflationZoneGrid[i][j]:
            colorImageNpArray[i][j][0] = 255
            colorImageNpArray[i][j][1] = 255
            colorImageNpArray[i][j][2] = 90
          else:
            colorImageNpArray[i][j][0] = 255
            colorImageNpArray[i][j][1] = 255
            colorImageNpArray[i][j][2] = 255

    if scaleFactor != 1:
      colorImageNpArray = colorImageNpArray.repeat(scaleFactor, axis=0).repeat(scaleFactor, axis=1)

    return colorImageNpArray

  def registerCorrectTrajOnMap(self, trajectory: trajectoryType):
    self.colorImageNpArrayWithTrajPlotted = np.copy(self.colorImageNpArray)
    print(f"LENGTH : {len(trajectory)}")
    for point in trajectory:

      x = int(point[0] / MAPSCALE)
      y = int(point[1] / MAPSCALE)
     
      # if y < 0 or y >= self.colorImageNpArrayWithTrajPlotted.shape[0] or x < 0 or x >= self.colorImageNpArrayWithTrajPlotted.shape[1]:
      self.colorImageNpArrayWithTrajPlotted[y][x] = np.array([0, 255, 0])

  def getColorImageWithTrajPlotted(self) -> np.ndarray:
    return self.colorImageNpArrayWithTrajPlotted

class CellularAutomataMapGenerator:
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

  def generateMap(
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


class Section:
  def __init__(self, startX: int, startY: int, width: int, height: int):
    self.startX = startX
    self.startY = startY
    self.width = width
    self.height = height
    self.centerX = int(startX + width / 2)
    self.centerY = int(startY + height / 2)
    self.lChild = None
    self.rChild = None

  def setLChild(self, lChild: Section):
    self.lChild = lChild

  def setRChild(self, rChild: Section):
    self.rChild = rChild

class MapBSPTree:
  def __init__(self, width: int, height: int, numIters: int):
    self.root = Section(0, 0, width, height)
    self.recursiveSplitSections(self.root, numIters)

  def randomSplit(self, section: Section) -> Tuple[Section, Section]: # direction, split coordinate
    lChild = None
    rChild = None
    lRatio = 0
    rRatio = 0
    validSplit = False

    while (not validSplit):
      if random.random() < 0.5: # Vertical split
        splitRangeDiv2 = int(section.height / 3)
        splitCoordinate = random.randint(section.centerY - splitRangeDiv2, section.centerY + splitRangeDiv2)
        lChildHeight = splitCoordinate - section.startY
        rChildHeight = section.height - lChildHeight
        
        lChild = Section(section.startX, section.startY, section.width, lChildHeight)
        lRatio = section.width/lChildHeight
        rChild = Section(section.startX, splitCoordinate, section.width, rChildHeight)
        rRatio = section.width/rChildHeight
      else: # Horizontal split
        splitRangeDiv2 = int(section.width / 3)
        splitCoordinate = random.randint(section.centerX - splitRangeDiv2, section.centerX + splitRangeDiv2)
        lChildWidth = splitCoordinate - section.startX
        rChildWidth = section.width - lChildWidth

        lChild = Section(section.startX, section.startY, lChildWidth, section.height)
        lRatio = section.height/lChildWidth
        rChild = Section(splitCoordinate, section.startY, rChildWidth, section.height)
        rRatio = section.height/rChildWidth

      if not (lRatio < 0.5 or lRatio > 2.5 or rRatio < 0.5 or rRatio > 2.5):
        validSplit = True
    
    return (lChild, rChild)

  def recursiveSplitSections(self, section: Section, numIters: int):
    if numIters == 0:
      return

    lChild, rChild = self.randomSplit(section)
    self.recursiveSplitSections(lChild, numIters - 1)
    self.recursiveSplitSections(rChild, numIters - 1)
    section.setLChild(lChild)
    section.setRChild(rChild)

class BSPTreeMapGenerator:
  def __init__(self):
    pass

  def spawnCorridor(self, lChild: Section, rChild: Section, mapGrid: np.ndarray):
    start = (lChild.centerY, lChild.centerX)
    end = (rChild.centerY + 2, rChild.centerX + 2)
    rr, cc = rectangle(start, end = end, shape=mapGrid.shape)
    mapGrid[rr, cc] = False

  def spawnRoom(self, section: Section, mapGrid: np.ndarray):
    roomWidth = random.randint(int(section.width * 0.5), int(section.width * 0.7))
    roomHeight = random.randint(int(section.height * 0.5), int(section.height * 0.7))

    roomWidthD2 = int(roomWidth/2)
    roomHeightD2 = int(roomHeight/2)

    start = (section.centerY - roomHeightD2, section.centerX - roomWidthD2)
    end = (section.centerY + roomHeightD2, section.centerX + roomWidthD2)

    rr, cc = rectangle(start, end = end, shape=mapGrid.shape)
    mapGrid[rr, cc] = False

  def spawnMap(self, rootSection: Section, mapGrid: np.ndarray):
    if rootSection.lChild != None:
      self.spawnCorridor(rootSection.lChild, rootSection.rChild, mapGrid)
      self.spawnMap(rootSection.lChild, mapGrid)
      self.spawnMap(rootSection.rChild, mapGrid)
    else:
      self.spawnRoom(rootSection, mapGrid)

  def generateMap(
    self, 
    mapWidth: int, 
    mapHeight: int, 
    scaleFactor: int,
    numIters: int
  ):
    width = int(mapWidth / scaleFactor)
    height = int(mapHeight / scaleFactor)
    mapBSTTree = MapBSPTree(width, height, numIters)
    mapGrid = np.ones((height, width), dtype = bool)
    self.spawnMap(mapBSTTree.root, mapGrid)

    # Upscale the generated map grid
    scaledMapGrid = mapGrid.repeat(scaleFactor, axis=0).repeat(scaleFactor, axis=1)
    return Map(scaledMapGrid)