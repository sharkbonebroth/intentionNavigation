from os import listdir
from os.path import isfile, join
from map import Map
import pickle
from typing import List, Tuple

class TrainingDataPoint():
  def __init__(self, startPoint, endPoint, direction, mapName, trajectory):
    self.startPoint = startPoint
    self.endPoint = endPoint
    self.direction = direction
    self.mapName = mapName
    self.trajectory = trajectory

class DataLoader:
  def __init__(self, mapDirPath: str, labelledDataDirPath: str, getDataMethod: str = "cyclic"):
    self.allMaps = {}
    self.allLabelledData = []
    self.loadAllMaps(mapDirPath)
    self.loadAllLabelledData(labelledDataDirPath)
    self.getDataMethod = getDataMethod
    self.currDataId = 0

  def getAllFilesInDir(self, dirPath: str):
    return [join(dirPath, f) for f in listdir(dirPath) if isfile(join(dirPath, f))]

  def loadAllMaps(self, mapDirPath: str):
    allMapFiles = self.getAllFilesInDir(mapDirPath)
    for mapFilePath in allMapFiles:
      print(mapFilePath)
      newMap = Map.fromFile(mapFilePath)
      self.allMaps[newMap.name] = newMap

  def loadSingleLabelledDataFile(self, labelledDataFilePath: str) -> List[TrainingDataPoint]:
    file = open(labelledDataFilePath, 'rb')
    trainingDataPoints = pickle.load(file)
    file.close()
    return trainingDataPoints

  def loadAllLabelledData(self, labelledDataDirPath: str):
    allLabelledDataFiles = self.getAllFilesInDir(labelledDataDirPath)
    for labelledDataFilePath in allLabelledDataFiles:
      self.allLabelledData.extend(self.loadSingleLabelledDataFile(labelledDataFilePath))

  def getLabelledDataAndMap(self) -> Tuple[TrainingDataPoint, Map]:
    labelledData, robotMap = self.getLabelledDataAndMapAtId(self.currDataId)
    self.currDataId += 1
    if self.currDataId == len(self.allLabelledData):
      self.currDataId = 0
    return labelledData, robotMap
  
  def getLabelledDataAndMapAtId(self, id) -> Tuple[TrainingDataPoint, Map]:
    labelledData = self.allLabelledData[id]
    return labelledData, self.allMaps[labelledData.mapName]