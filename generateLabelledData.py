import tkinter as tk
import tkinter.font as font
from PIL import Image, ImageTk
from map import *
from path import *
from skimage.draw import disk, line
import time
import argparse
import sys
import pickle
import math
import copy
import random
from pathlib import Path
from typing import List, Tuple
from dataLoader import TrainingDataPoint

parser = argparse.ArgumentParser(
    prog = "main.py",
    description= "Code for generating labelled data"
)

parser.add_argument("-mapDir", type = str, default = "maps", help = "Directory to save generated maps to")
parser.add_argument("-labelledDataDir", type = str, default = "labelledData")
args = None

class DataLabelingStateMachine():
  def __init__(self):
    self.state = 0

  def reset(self):
    self.state = 0

  def next(self):
    if self.state == 0:
      self.state = 1
    elif self.state == 1:
      self.state = 2
    else:
      self.state = 0

  def getState(self):
    return self.state

  def getStateString(self):
    if self.state == 0:
      return "labelling start point"
    elif self.state == 1:
      return "labelling end point"
    else:
      return "confirming start and end points"

mapGenCA = CellularAutomataMapGenerator()
mapGenBSP = BSPTreeMapGenerator()
def generateMap():
  if random.random() > 0.5:
    return mapGenCA.generateMap(1200, 800, 16, "random", 0.5, 5, 4, 7)
  else:
    return mapGenBSP.generateMap(1200, 800, 16, 4)

def updateMapImage(imgNpArray: np.ndarray):
  global panel
  imgNew =  ImageTk.PhotoImage(image=Image.fromarray(imgNpArray))
  panel.configure(image = imgNew)
  panel.image = imgNew

def toggleDataLabelingButtons(state: bool):
    global conformStartAndEndPointsButtonLeft, conformStartAndEndPointsButtonStraight, conformStartAndEndPointsButtonRight
    if state:
      conformStartAndEndPointsButtonLeft["state"] = "normal"
      conformStartAndEndPointsButtonStraight["state"] = "normal"
      conformStartAndEndPointsButtonRight["state"] = "normal"
    else:
      conformStartAndEndPointsButtonLeft["state"] = "disabled"
      conformStartAndEndPointsButtonStraight["state"] = "disabled"
      conformStartAndEndPointsButtonRight["state"] = "disabled"


def generateMapCallback():
  toggleDataLabelingButtons(False)
  global map, imgNpArray, originalImgNpArray, startPoint, endPoint, dataLabelingStateMachine, mapSaved, trajectory
  map = generateMap()
  mapSaved = False
  imgNpArray = map.getColorImageNpArray()
  originalImgNpArray = copy.deepcopy(imgNpArray)

  updateMapImage(imgNpArray)
  startPoint = None
  endPoint = None
  trajectory = None
  dataLabelingStateMachine.reset()

def plotPoint(x: int, y: int, npArrayImg: np.ndarray, pixelValue: np.ndarray):
  rr, cc = disk(center = (y, x), radius = 5)
  for i in rr:
    for j in cc:
      npArrayImg[i][j] = pixelValue

def plotLine(xStart: int, yStart: int, angle: float, npArrayImg: np.ndarray, pixelValue: np.ndarray, lineLength = 50):
  xEnd = xStart + int(math.cos(angle) * lineLength)
  yEnd = yStart + int(math.sin(angle) * lineLength)
  
  rr, cc = line(yStart, xStart, yEnd, xEnd)
  npArrayImg[rr, cc] = pixelValue

def plotTrajectory(trajectory: List[Tuple[int, int]], npArrayImg: np.ndarray, pixelValue: np.ndarray):
  for point in trajectory:
    npArrayImg[point[1]][point[0]] = pixelValue

def mouseClickCallback(eventOrigin):
  global imgNpArray
  widgetName = str(eventOrigin.widget).split('.')[-1]
  if (widgetName == "map"):
    x = eventOrigin.x
    y = eventOrigin.y

    if (dataLabelingStateMachine.getState() == 0) and map.mapGrid[y][x] == False:
      global startPoint 
      startPoint = [x, y, 0]
      plotPoint(x, y, imgNpArray, [0, 0, 255])
    elif (dataLabelingStateMachine.getState() == 1) and map.mapGrid[y][x] == False:
      global endPoint
      endPoint = [x, y, 0]
      plotPoint(x, y, imgNpArray, [255, 0, 0])
      toggleDataLabelingButtons(True)

    updateMapImage(imgNpArray)

def mouseReleaseCallback(eventOrigin):
  global imgNpArray
  global trajectory
  widgetName = str(eventOrigin.widget).split('.')[-1]
  if (widgetName == "map"):
    x = eventOrigin.x
    y = eventOrigin.y

    global startPoint, endPoint
    if (dataLabelingStateMachine.getState() == 0) and startPoint != None:
      xDiff = x - startPoint[0]
      yDiff = y - startPoint[1]
      angle = math.atan2(yDiff, xDiff)
      startPoint[2] = angle
      plotLine(startPoint[0], startPoint[1], angle, imgNpArray, [0, 255, 0])
      dataLabelingStateMachine.next()

    elif (dataLabelingStateMachine.getState() == 1 and endPoint != None):
      xDiff = x - endPoint[0]
      yDiff = y - endPoint[1]
      angle = math.atan2(yDiff, xDiff)
      endPoint[2] = angle
      plotLine(endPoint[0], endPoint[1], angle, imgNpArray, [0, 255, 0])
      solver = AStarTrajectorySolver()
      traj = solver.solveTrajectory(map, startPoint, endPoint)
      trajectory = copy.deepcopy(traj)
      plotTrajectory(traj, imgNpArray, [0, 255, 0])
      toggleDataLabelingButtons(True)
      dataLabelingStateMachine.next()

    updateMapImage(imgNpArray)


def resetImgNpArray():
  global imgNpArray, originalImgNpArray
  imgNpArray = copy.deepcopy(originalImgNpArray)


def confirmStartAndEndPointsCallback(direction: int): # -1: left, 0: straight, 1: right
  global startPoint, endPoint, trajectory

  if dataLabelingStateMachine.getState() == 2:
    global map, mapSaved, args
    if not mapSaved:
      mapName = time.strftime('%Y%m%d_%H%M%S')
      map.setName(mapName)
      Path(args.mapDir).mkdir(parents=True, exist_ok=True)
      savePath = f"{args.mapDir}/{mapName}"
      map.saveToFile(savePath)
      mapSaved = True

    print("New start and end points confirmed")

    startPointConvertedToDist = (startPoint[0] * MAPSCALE, startPoint[1] * MAPSCALE, startPoint[2])
    endPointConvertedToDist = (endPoint[0] * MAPSCALE, endPoint[1] * MAPSCALE, endPoint[2])
    trajectoryConvertedToDist = [(pt[0] * MAPSCALE, pt[1] * MAPSCALE) for pt in trajectory]
    TrainingDataPoints.append(
      TrainingDataPoint(
        startPointConvertedToDist,
        endPointConvertedToDist,
        direction,
        map.name,
        trajectoryConvertedToDist
      )
    )
    resetImgNpArray()
    startPoint = None
    endPoint = None
    trajectory = None
    dataLabelingStateMachine.next()

    toggleDataLabelingButtons(False)
    updateMapImage(imgNpArray)

def cancelLabelling():
  dataLabelingStateMachine.reset()

  global startPoint, endPoint, trajectory
  startPoint = None
  endPoint = None
  trajectory = None

  resetImgNpArray()
  toggleDataLabelingButtons(False)
  updateMapImage(imgNpArray)

def finishLabelling(eventOrigin):
  global TrainingDataPoints
  print("Saving labelled data...")
  labelledDataFileName = time.strftime('%Y%m%d_%H%M%S')
  Path(args.labelledDataDir).mkdir(parents=True, exist_ok=True)
  savePath = f"{args.labelledDataDir}/{labelledDataFileName}"
  pickle.dump(TrainingDataPoints, open(savePath, 'wb'))
  
  app.withdraw()
  sys.exit()

# Initialize the data label array, and the labelling state machine
TrainingDataPoints = []
startPoint = None
endPoint = None
trajectory = None
dataLabelingStateMachine = DataLabelingStateMachine()

# Initialize the GUI
app = tk.Tk()
app.geometry("1920x1080")
frame = tk.Frame()
frame.pack()

# Initialize the initial map
mapSaved = False
map = generateMap()

# map = mapGen.generateMap(1200, 800, 16, "random", 0.5, 5, 4, 7)
imgNpArray = map.getColorImageNpArray()
originalImgNpArray = copy.deepcopy(imgNpArray)
img =  ImageTk.PhotoImage(image=Image.fromarray(imgNpArray))
panel = tk.Label(app, image=img, name="map")
panel.pack()


# BUTTONS
generateMapButton = tk.Button(
  master=frame,
  text="Generate Map!",
  width=25,
  height=2,
  bg="grey",
  fg="yellow",
  command = generateMapCallback,
  font = font.Font(size=15)
)

entry = tk.Entry(master=frame, fg="yellow", bg="grey", width=20, font = font.Font(size=15))

conformStartAndEndPointsButtonLeft = tk.Button(
  master=frame,
  text="Left",
  width=25,
  height=2,
  bg="grey",
  fg="yellow",
  command = lambda: confirmStartAndEndPointsCallback(-1),
  font = font.Font(size=15)
)

conformStartAndEndPointsButtonStraight = tk.Button(
  master=frame,
  text="Straight",
  width=25,
  height=2,
  bg="grey",
  fg="yellow",
  command = lambda: confirmStartAndEndPointsCallback(0),
  font = font.Font(size=15)
)

conformStartAndEndPointsButtonRight = tk.Button(
  master=frame,
  text="Right",
  width=25,
  height=2,
  bg="grey",
  fg="yellow",
  command = lambda: confirmStartAndEndPointsCallback(1),
  font = font.Font(size=15)
)

cancelLabellingButton = tk.Button(
  master=frame,
  text="Cancel Labelling",
  width=25,
  height=2,
  bg="grey",
  fg="yellow",
  command = cancelLabelling,
  font = font.Font(size=15)
)

generateMapButton.pack(side = tk.LEFT, pady = 25)
conformStartAndEndPointsButtonLeft.pack(side = tk.LEFT, pady = 25)
conformStartAndEndPointsButtonStraight.pack(side = tk.LEFT, pady = 25)
conformStartAndEndPointsButtonRight.pack(side = tk.LEFT, pady = 25)
cancelLabellingButton.pack(side = tk.LEFT, pady = 25)
toggleDataLabelingButtons(False)


app.bind("<Button 1>", mouseClickCallback)
app.bind("<ButtonRelease-1>", mouseReleaseCallback)
app.bind('<Escape>', finishLabelling)

if __name__ == "__main__":
  args = parser.parse_args()
  app.mainloop()


  