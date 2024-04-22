from typing import Tuple, List
import numpy as np
from map import Map
from scipy import ndimage
import math
from utilTypes import Action

odometryDataPointType = Tuple[float, float] # delta x, delta y
MAPSCALE = 0.05 # each pixel is 0.05m

# Class implementing the sensor feedback interface for feeding into the planner.
class Robot:
  def __init__(self, startX: float, startY: float, yaw: float, map: Map):
    self.odometry = [] # Odometry with gaussian noise added; In world frame
    self.currPositionActual = (startX,startY,yaw) # keep track of robot current position; no gaussian noise; In world frame
    self.currPositionEstimate = (startX,startY,yaw) # in world framne
    self.dt = 0.1 
    self.map = map
  
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
    odomToPlot = self.getNLatestOdometries(100)
    for odom in odomToPlot: # loop will miss the last one but wtv
      dx, dy = odom

      xBeingPlottedDiscretized = int(xBeingPlotted / MAPSCALE)
      yBeingPlottedDiscretized = int(yBeingPlotted / MAPSCALE)

      mapWithOdomPlottedWorldFrame[yBeingPlottedDiscretized][xBeingPlottedDiscretized] = np.array([255,0,0])

      xBeingPlotted = xActual - dx
      yBeingPlotted = yActual - dy  

    # Crop the lidar reading image out
    lidarRangeCroppedPx = math.floor(5/MAPSCALE)
    lidarRangeUncroppedPx = np.ceil(lidarRangeCroppedPx * math.sqrt(2)) #5m vision range, account for potential cropping later
    lidarImgUncroppedSize = 2 * lidarRangeUncroppedPx + 1
    lidarImgUncropped = np.zeros(lidarImgUncroppedSize, lidarImgUncroppedSize, 3)
    xActualImgCoord = int(xActual/MAPSCALE)
    yActualImgCoord = int(yActual/MAPSCALE)
    for yCoordLidarImg, yCoordOdomImg in enumerate(range(yActualImgCoord - lidarRangeUncroppedPx, yActualImgCoord + lidarRangeUncroppedPx + 1)):
      for XxoordLidarImg, xCoordOdomImg in enumerate(range(xActualImgCoord - lidarRangeUncroppedPx, xActualImgCoord + lidarRangeUncroppedPx + 1)):
        if yCoordOdomImg < 0 or yCoordOdomImg >= self.map.height or xCoordOdomImg < 0 or xCoordOdomImg >= self.map.width:
          continue
        else:
          lidarImgUncropped[yCoordLidarImg][XxoordLidarImg] = mapWithOdomPlottedWorldFrame[yCoordOdomImg][xCoordOdomImg]

    # rotate and Crop the uncropped lidar image in the center
    lidarImgUncroppedRotated = ndimage.rotate(lidarImgUncropped, yawActual, reshape=True)
    centerX = lidarRangeUncroppedPx + 1
    centerY = centerX

    startX = centerX - lidarRangeCroppedPx
    endX = centerX + lidarRangeCroppedPx
    startY = startX
    endY = endX
    
    lidarImg = lidarImgUncroppedRotated[startY:endY, startX:endX]
    return lidarImg

  #get the latest n odometry points
  def getNLatestOdometries(self, n: int):
    return self.odometry[len(self.odometry)-n:]

  def getRotationMatrix(self, angle:float):
    angle_rad = np.deg2rad(angle)
    return np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
  
  def isLegalMovement(self, coord):
    map = self.getMap()
    x, y = coord
    h, w = map.shape[:2]
    inBound: bool = (0<=x<w) and (0<=y<h)
    collisionFree: bool = (map[y,x,:]==np.array([255,255,255]))
    return (inBound and collisionFree)

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
    linXDivOmega = linX/omega
    dx = linXDivOmega * math.sin(omega * self.dt)
    dy = linXDivOmega - linXDivOmega * math.cos(omega * self.dt)

    return dx, dy

  def move(self, action: Action):
    #note: Assume positive angle to be clockwise. This is because we are using image coordinates
    linX, omega = action.get_as_ndarray() #action is in robot frame
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
    dxRobotFrameNoised = self.addGaussianNoise(dxRobotFrame, 0.05)
    dyRobotFrameNoised = self.addGaussianNoise(dyRobotFrame, 0.05)
    dYawNoised = self.addGaussianNoise(dYaw, 0.05)
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
