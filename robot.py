from typing import Tuple, List
import numpy as np
from map import Map
from scipy import ndimage
import math
from ppo import Action

odometryDataPointType = Tuple[float, float] # delta x, delta y
MAPSCALE = 0.05 # each pixel is 0.05m

# Class implementing the sensor feedback interface for feeding into the planner.
class Robot:
  def __init__(self, startX: float, startY: float, yaw: float):
    self.odometry = [(startX,startY,yaw)]
    self.curr_pos = (startX,startY,yaw) #keep track of robot current position
  
  # As it is expensive to simulate the movement of the robot directly, we assume that the robot can reach the way point
  # generated by the planner, and generate some form of odometry while travelling there. Gaussian noise is then added for
  # robustness 
  '''
  Question: should Odometry be a list of poses in world frame, or delta movement? right now it is the delta movement
  Qeustion2: if a list of poses in world frame, not so meaningful to add noise, noise should be added in the rotation and displacement stage?
  '''
  def addToOdometryWithGaussianNoise(self, startPoint: Tuple[float, float], endPoint: Tuple[float, float], variance = 0.05):
    dx = endPoint[0] - startPoint[0] + np.random.normal(0, variance)
    dy = endPoint[1] - startPoint[1] + np.random.normal(0, variance)
    self.odometry.append(dx, dy)

  def addToOdometry(self, point:Tuple[float,float,float]):
    self.odometry.append(point)

  def getMap(self) -> Map:
    '''
    TO DO: implement a function to get current map. How to? Which function should I call?
    '''
    return

  #return lidar image in robot frame, with robot at the center
  def getLidarReadingImage(self, coord) -> np.ndarray:
    map = self.getMap()
    x, y = coord[0], coord[1]
    lidar_range = math.floor(5/MAPSCALE) #5m vision range
    lidar_image = np.zeros([lidar_range,lidar_range,3])
    '''
    TO DO: technically, once a FOV is blocked by obstacle, any pixel behind it should be UNKNOWN instead.
    '''
    for i in range (y-lidar_range,y+lidar_range+1):
      for j in range(x-lidar_range,x+lidar_range+1):
        if ((i==y) and (j==x)) : continue
        lidar_image[i,j,:] = map[i,j,:]
    return lidar_image

  #get the latest n odometry points
  def getLatestOdometry(self, n: int):
    return self.odometry[len(self.odometry)-n:]

  #get the current robot pose: location & yaw
  def getCurrentRobotPosWorld(self):
    return self.getLatestOdometry(1)

  def plotOdometryOnLidarReadingImage(self, lidarReadingImage: np.ndarray) -> np.ndarray:
    '''
    TO DO: need to plot yaw as well? right now I am not
    '''
    h, w = lidarReadingImage.shape[:2]
    odom = self.getLatestOdometry(20) #the last 20 odometry points
    lidar_odom_image = lidarReadingImage.copy()
    for i in range(len(odom)):
      x, y, _ = odom[i]
      if not ((0<=x<w) and (0<=y<h)): continue
      lidar_odom_image[y,x,:] = np.array([255,0,0]) #odom is plotted with (255,0,0): RED
    return lidar_odom_image

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

  def _move(self,dx,dy):
    move_x, move_y = dx, dy
    curr_x, curr_y = self.getCurrentRobotPosWorld()
    target_x, target_y = curr_x+move_x, curr_y+move_y
    while not self.isLegalMovement(target_x,target_y):
      if (move_x!=0 and move_y!=0):
        move_y = move_y-1
        move_x = math.ceil(move_x - (move_x/move_y)) #similar triangle. use math.ceil to adapt to discrete grid map. will result in movemment vector differ from original.
        target_x, target_y = curr_x+move_x, curr_y+move_y
      elif move_y==0: #straight movement in x-direction
        move_x -= move_x
        target_x = curr_x+move_x
      elif move_x==0: #straight movment in y-direction
        move_y -= move_y
        target_y = curr_y+move_y
    return (target_x, target_y)

  def move(self, action: Action):
    #note: Assume positive angle to be counter-clockwise
    dt = 1.0 #simulation time (1.0 second is easy to implement, any angle that one wishes to implement, can just use as ang_z, same thing with displacement)
    lin_x, ang_z = action.get_as_ndarray() #action is in robot frame
    theta = ang_z * dt #assume robot motion is turn first, then only move linearly (simpler case than omni-directional robot where turning and moving happen at same time)
    dispRobot = lin_x * dt #net displacement in robot frame
    prev_yaw = self.getCurrentRobotPosWorld()[-1]
    yaw = theta + prev_yaw #net angle from world frame, in degree
    rotationMatrix = self.getRotationMatrix(yaw)
    dispWorld = np.matmul(rotationMatrix, np.array([1,0])) * dispRobot #displacement in world frame
    dx, dy = list(map(lambda x: math.ceil(x/MAPSCALE), dispWorld)) #how many pixels the robot is commanded to move
    final_x, final_y = self._move(dx,dy)
    self.addToOdometry((final_x,final_y,yaw))
 
  #rotate an image by angle about center of the image
  def rotateImage(I:np.ndarray, angle:float) -> np.ndarray:
    return ndimage.rotate(I,angle,reshape=True)

  def getFeedbackImage(self):
    angle = self.getLatestOdometry(1)[-1]
    lidar_image = self.getLidarReadingImage(self.curr_pos)
    lidar_odom_image = self.plotOdometryOnLidarReadingImage(lidar_image)
    return self.rotateImage(lidar_odom_image, -angle) #rotate in opposite direction to robot's heading

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
