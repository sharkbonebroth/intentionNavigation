import tkinter as tk
import tkinter.font as font
from PIL import Image, ImageTk
from map import *
from skimage.draw import disk
import time
import argparse

parser = argparse.ArgumentParser(
    prog = "main.py",
    description= "Code for demonstrating QLearning, First Visit Monte Carlo and SARSA"
)

parser.add_argument("-mapDir", type = str, default = "maps", help = "Directory to save generated maps to")

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

def updateMapImage(imgNpArray: np.ndarray):
  global panel
  imgNew =  ImageTk.PhotoImage(image=Image.fromarray(imgNpArray))
  panel.configure(image = imgNew)
  panel.image = imgNew

def generateMapCallback():
  global map, imgNpArray, startPoint, endPoint, dataLabelingStateMachine, mapSaved
  map = mapGen.generateMapCellularAutomata(1200, 800, 16, "random", 0.5, 5, 4, 5)
  mapSaved = False
  imgNpArray = map.getColorImageNpArray()
  updateMapImage(imgNpArray)
  startPoint = None
  endPoint = None
  dataLabelingStateMachine.reset()

def plotPoint(x: int, y: int, npArrayImg: np.ndarray, pixelValue: np.ndarray):
  rr, cc = disk(center = (y, x), radius = 5)
  for i in rr:
    for j in cc:
      npArrayImg[i][j] = pixelValue

def mouseClickCallback(eventorigin):
  global imgNpArray
  widgetName = str(eventorigin.widget).split('.')[-1]
  if (widgetName == "map"):
    x = eventorigin.x
    y = eventorigin.y

    if (dataLabelingStateMachine.getState() == 0) and map.mapGrid[y][x] == False:
      global startPoint 
      startPoint = (x, y)
      plotPoint(x, y, imgNpArray, [0, 0, 255])
      dataLabelingStateMachine.next()
    elif (dataLabelingStateMachine.getState() == 1) and map.mapGrid[y][x] == False:
      global endPoint
      endPoint = (x, y)
      plotPoint(x, y, imgNpArray, [255, 0, 0])
      dataLabelingStateMachine.next()

    updateMapImage(imgNpArray)

def confirmStartAndEndPointsCallback():
  if dataLabelingStateMachine.getState() == 2:
    global map, mapSaved
    if not mapSaved:
      map.saveToFile(time.strftime('%Y%m%d_%H%M%S'))
      mapSaved = True

    print("New start and end points confirmed")
    startAndEndPoints.append((startPoint, endPoint))
    plotPoint(startPoint[0], startPoint[1], imgNpArray, [255, 255, 255])
    plotPoint(endPoint[0], endPoint[1], imgNpArray, [255, 255, 255])
    dataLabelingStateMachine.next()    

    updateMapImage(imgNpArray)


# Initialize the data label array, and the labelling state machine
startAndEndPoints  = []
startPoint = None
endPoint = None
dataLabelingStateMachine = DataLabelingStateMachine()

# Initialize the GUI
app = tk.Tk()
app.geometry("1500x1000")
frame = tk.Frame()
frame.pack()

# Initialize the initial map
mapSaved = False
mapGen = mapGenerator()
map = mapGen.generateMapCellularAutomata(1200, 800, 16, "random", 0.5, 5, 4, 5)
imgNpArray = map.getColorImageNpArray()
img =  ImageTk.PhotoImage(image=Image.fromarray(imgNpArray))
panel = tk.Label(app, image=img, name="map")
panel.pack()

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

conformStartAndEndPointsButton = tk.Button(
  master=frame,
  text="Confirm labels",
  width=25,
  height=2,
  bg="grey",
  fg="yellow",
  command = confirmStartAndEndPointsCallback,
  font = font.Font(size=15)
)

generateMapButton.pack(side = tk.LEFT, pady = 25)
conformStartAndEndPointsButton.pack(side = tk.LEFT, pady = 25)
entry.pack(side = tk.RIGHT, pady = 25, padx = 10)


app.bind("<Button 1>",mouseClickCallback)
app.mainloop()


  