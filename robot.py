from typing import Tuple, List
import numpy as np
from map import Map
from scipy import ndimage
import math
from utilTypes import Action

odometryDataPointType = Tuple[float, float] # delta x, delta y
MAPSCALE = 0.05 # each pixel is 0.05m
LIDARRANGE = 7
FOV_SIDE_SIZE = int(LIDARRANGE/MAPSCALE + 1)
FOV_SIZE = (FOV_SIDE_SIZE, FOV_SIDE_SIZE) # LIDARRANGE / MAPSCALE * 2 + 1

# Class implementing the sensor feedback interface for feeding into the planner.
class Robot:
  def __init__(self, startX: float, startY: float, yaw: float, map: Map, numOdomToPlot: int = 200, noiseVariance: float = 0.005):
    self.odometry = [] # Odometry with gaussian noise added; In world frame
    self.currPositionActual = (startX,startY,yaw) # keep track of robot current position; no gaussian noise; In world frame
    self.currPositionEstimate = (startX,startY,yaw) # in world framne
    self.dt = 0.1 
    self.map = map
    self.numOdomToPlot = numOdomToPlot
    self.noiseVariance = noiseVariance
    
  def reset(self, startX: float, startY: float, yaw: float):
    self.currPositionActual = (startX,startY,yaw) # keep track of robot current position; no gaussian noise; In world frame
    self.currPositionEstimate = (startX,startY,yaw) # in world framne
  
  def addGaussianNoise(self, value: float, variance: float = 0.05):
    return value + np.random.normal(0, variance)

  #return lidar image in robot frame, with robot at the center, transformed into robotFrame
  def getFeedbackImage(self) -> np.ndarray:
    xActual, yActual, yawActual = self.currPositionActual

    # Initialize the image for plotting
    mapWithOdomPlottedWorldFrame = np.copy(self.map.colorImageNpArray)

    # Plot the odometry estimates on the color mapgrid
    xBeingPlotted = xActual
    yBeingPlotted = yActual
    odomToPlot = self.getNLatestOdometries(self.numOdomToPlot)
    for odom in odomToPlot: # loop will miss the last one but wtv
      dx, dy = odom

      xBeingPlottedDiscretized = int(xBeingPlotted / MAPSCALE)
      yBeingPlottedDiscretized = int(yBeingPlotted / MAPSCALE)

      mapWithOdomPlottedWorldFrame[yBeingPlottedDiscretized][xBeingPlottedDiscretized] = np.array([255,0,0])

      xBeingPlotted = xBeingPlotted - dx
      yBeingPlotted = yBeingPlotted - dy  

    # Crop the lidar reading image out
    lidarRangeCroppedPx = math.floor(LIDARRANGE/MAPSCALE)
    lidarRangeUncroppedPx = int(np.ceil(lidarRangeCroppedPx * math.sqrt(2))) #5m vision range, account for potential cropping later
    lidarImgUncroppedSize = int(2 * lidarRangeUncroppedPx + 1)
    lidarImgUncropped = np.zeros((lidarImgUncroppedSize, lidarImgUncroppedSize, 3), dtype=np.uint8)
    xActualImgCoord = int(xActual/MAPSCALE)
    yActualImgCoord = int(yActual/MAPSCALE)
    for yCoordLidarImg, yCoordOdomImg in enumerate(range(yActualImgCoord - lidarRangeUncroppedPx, yActualImgCoord + lidarRangeUncroppedPx + 1)):
      for XCoordLidarImg, xCoordOdomImg in enumerate(range(xActualImgCoord - lidarRangeUncroppedPx, xActualImgCoord + lidarRangeUncroppedPx + 1)):
        if yCoordOdomImg < 0 or yCoordOdomImg >= self.map.height or xCoordOdomImg < 0 or xCoordOdomImg >= self.map.width:
          continue
        else:
          lidarImgUncropped[yCoordLidarImg][XCoordLidarImg] = mapWithOdomPlottedWorldFrame[yCoordOdomImg][xCoordOdomImg]

    # rotate and Crop the uncropped lidar image in the center
    lidarImgUncroppedRotated = ndimage.rotate(lidarImgUncropped, yawActual*180/np.pi, reshape=False)
    centerX = lidarRangeUncroppedPx + 1

    startX = centerX - lidarRangeCroppedPx
    endX = centerX + lidarRangeCroppedPx + 1
    startY = startX
    endY = endX
    
    lidarImg = lidarImgUncroppedRotated[startY:endY, startX:endX]
    return lidarImg
    
  #get the latest n odometry points
  def getNLatestOdometries(self, n: int):
    if n > len(self.odometry):
      n = len(self.odometry)
    return self.odometry[::-1][:n]

  def hasCrashedIntoWall(self) -> bool:
    mapGrid = self.map.mapGrid
    x = self.currPositionActual[0]
    y = self.currPositionActual[1]
    xDiscretized = int(x / MAPSCALE)
    yDiscretized = int(y / MAPSCALE)
    # We assume the edge of the map is a wall too!
    if (x < 0) or (x > MAPSCALE * mapGrid.width) or (y < 0) or (y > MAPSCALE * mapGrid.height):
      return True
    return mapGrid[yDiscretized][xDiscretized]
    
  def isInInflationZone(self) -> bool:
    x = self.currPositionActual[0]
    y = self.currPositionActual[1]
    xDiscretized = int(x)
    yDiscretized = int(y)
    return self.map.inflationZoneGrid[yDiscretized][xDiscretized]

  def getDisplacementRobotFrame(self, linX, omega):
    if omega == 0:
      return linX * self.dt, 0
    linXDivOmega = linX/omega
    dx = linXDivOmega * math.sin(omega * self.dt)
    dy = linXDivOmega - linXDivOmega * math.cos(omega * self.dt)

    return dx, dy

  def move(self, action: Action):
    #note: Assume positive angle to be clockwise. This is because we are using image coordinates
    linX, omega = action.lin_x, action.ang_z #action is in robot frame
    dYaw = omega * self.dt
    dxRobotFrame, dyRobotFrame = self.getDisplacementRobotFrame(linX, omega)    
    
    # Update actual position of robot
    prevXWorldActual, prevYWorldActual, prevYawWorldActual = self.currPositionActual
    dxWorldFrameActual = dxRobotFrame * math.cos(prevYawWorldActual) - dyRobotFrame * math.sin(prevYawWorldActual)
    dyWorldFrameActual = dxRobotFrame * math.sin(prevYawWorldActual) + dyRobotFrame * math.cos(prevYawWorldActual)
    currXWorldActual = prevXWorldActual + dxWorldFrameActual
    currYWorldActual = prevYWorldActual + dyWorldFrameActual
    currYawWorldActual = prevYawWorldActual + dYaw
    self.currPositionActual = (currXWorldActual, currYWorldActual, currYawWorldActual) # Update the current position of the robot

    # Update noisy estimates
    prevXWorldEstimate, prevYWorldEstimate, prevYawWorldEstimate = self.currPositionEstimate
    dxRobotFrameNoised = self.addGaussianNoise(dxRobotFrame, self.noiseVariance)
    dyRobotFrameNoised = self.addGaussianNoise(dyRobotFrame, self.noiseVariance)
    dYawNoised = self.addGaussianNoise(dYaw, self.noiseVariance)
    dxWorldFrameEstimate = dxRobotFrameNoised * math.cos(prevYawWorldEstimate) - dyRobotFrameNoised * math.sin(prevYawWorldEstimate)
    dyWorldFrameEstimate = dxRobotFrameNoised * math.sin(prevYawWorldEstimate) + dyRobotFrameNoised * math.cos(prevYawWorldEstimate)
    currXWorldEstimate = prevXWorldEstimate + dxWorldFrameEstimate
    currYWorldEstimate = prevYWorldEstimate + dyWorldFrameEstimate
    currYawWorldEstimate = prevYawWorldEstimate + dYawNoised
    self.currPositionEstimate = (currXWorldEstimate, currYWorldEstimate, currYawWorldEstimate)

    # Add noisy odom to odometry
    self.odometry.append((dxWorldFrameEstimate, dyWorldFrameEstimate))

# Class implementating the low level controller, which will be used to control the robot to follow the waypoints. This 
# is not used in training (the simulated odometry with gaussian noise is used instead), but will be useful in evaluation
class RobotController:
  def __init__(self, robot: Robot, robotStartX: float, robotStartY: float, updateRate: int):
    self.robot = robot
    self.robotX = robotStartX,
    self.robotY = robotStartY
    self.updateRate = updateRate
    self.waypoints = []

  def addWaypoint(self, waypoint: Tuple[float, float]):
    pass

  def getControlInputs(self) -> Tuple[float, float]:
    pass

  # Moves the robot in the map based on the control inputs it generate (with added gaussian noise) and the updateRate, 
  # and updates the robot's odometry accordingly (also with added gaussian noise)
  def moveRobot(self):
    pass
