from robot import Robot
from utilTypes import Action, trajectoryType, find_closest_point, Intention, get_distance
from map import Map, MAPSCALE
from typing import Tuple, List
import matplotlib.pyplot as plt
import gymnasium
import numpy as np
import torch
from skimage.draw import disk, line
class Reward:
    CRASHING : float = -1.0
    ININFLATIONZONE : float = -0.5

class IntentionNavEnv(gymnasium.Env):
    MAX_STEPS = 10000
    def __init__(self, obs_space_shape : Tuple, pathsIn : List[trajectoryType], intentionsIn : List[Intention], mapIn : Map, startPoint, endPoint):
        self.done : bool = False
        self.obs_space_shape : tuple = obs_space_shape
        self.paths : List[trajectoryType] = pathsIn
        self.map : Map = mapIn
        self.robot = Robot(map=mapIn, startX=startPoint[0], startY=startPoint[1], yaw=startPoint[2])
        self.steps = 0
        self.prevRobotPoseWorld = self.robot.currPositionActual
        self.intentions = intentionsIn
        self.trainingId = 0
        
        self.startPoint = startPoint
        self.endPoint = endPoint
        
        self.curPath = self.paths[self.trainingId]
        self.curIntention = self.intentions[self.trainingId]
        
        self.curBestWaypointId = 0
        self.totalReward = 0

        plt.figure(figsize=(8, 6), dpi=150)
        
    def getObservations(self):
        return self.robot.getBinaryFeedbackImage(scaleFactor=4), float(self.curIntention)
        
    def step(self, action : np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self.robot.move(Action(*action)) 
        
        obs, intention = self.getObservations()
        
        curRobotPoseWorld = self.robot.getRobotPoseWorld()
        reward = self.get_reward(action, curRobotPoseWorld, self.prevRobotPoseWorld)
        done = self.is_done(curRobotPoseWorld)
            
        self.prevRobotPoseWorld = curRobotPoseWorld
        
        info = dict()
        self.totalReward += reward
        self.steps += 1
        info['episode'] = {
            'reward' : self.totalReward / self.steps,
            'length' : self.steps
        }
        return obs, intention, reward, done, info
    
    def get_reward(self, action : np.ndarray, curRobotPos, prevRobotPos):
        closestWaypointId = find_closest_point(curRobotPos[:2], self.curPath)
        
        inverse = 1
        if self.curBestWaypointId > closestWaypointId:
            inverse *= -1
        reward = inverse * get_distance(self.curPath[closestWaypointId], self.curPath[self.curBestWaypointId])
        
        if self.robot.hasCrashedIntoWall():
            reward += Reward.CRASHING
        elif self.robot.isInInflationZone():
            reward += Reward.ININFLATIONZONE
        
        if closestWaypointId > self.curBestWaypointId:
            self.curBestWaypointId = closestWaypointId
        return reward
    
    def reset(self):
        self.steps = 0
        self.totalReward = 0
        self.prevRobotPoseWorld = self.startPoint
        self.robot.reset(*self.startPoint)
        # if self.trainingId >= len(self.paths):
        #     return np.zeros((640,480))
        # self.trainingId +=1
        # self.curPath = self.paths[self.trainingId]
        # self.curIntention = self.intentions[self.trainingId]
    
    def is_done(self, curRobotPoseWorld):
        if self.robot.hasCrashedIntoWall():
            print("Crashed!!")
            return True
        if get_distance(curRobotPoseWorld, self.endPoint) < 0.1:
            return True
        if self.steps >= IntentionNavEnv.MAX_STEPS:
            return True
        return False
    
    
    def render(self):
        img = np.copy(self.robot.mapImgWithPerfectOdomPlotted)
        
        # Plot the current position of the robot
        robotPosition = self.robot.currPositionActual
        robotImgX = int(robotPosition[0] / MAPSCALE)
        robotImgY = int(robotPosition[1] / MAPSCALE)
        rr, cc = disk(center = (robotImgY, robotImgX), radius = 5)
        img[rr, cc] = np.array([0, 0, 255])

        # Plot its angle
        robotYaw = robotPosition[2]
        endX = int(np.cos(robotYaw) * 10) + robotImgX
        endY = int(np.sin(robotYaw) * 10) + robotImgY
        rr, cc = line(robotImgY, robotImgX, endY, endX)
        img[rr, cc] = np.array([0, 255, 0])

        plt.clf()
        plt.imshow(img)
        plt.draw()
        plt.pause(0.01)
    
class DummyIntentionNavEnv(gymnasium.Env):
    def __init__(self, obs_space_shape):
        self.done : bool = False
        self.obs_space_shape = obs_space_shape
        
    def step(self, action : np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs = self.get_observation()
        reward = self.get_reward(action)
        done = self.is_done()
        info = dict()
        return obs, reward, done, info
        
    def is_done(self) -> bool:
        """
        TODO: Returns episode completion state
        """
        return self.done
        
    def get_observation(self) -> np.ndarray:
        """
        TODO: Returns current observation
        """
        return torch.rand(self.obs_space_shape)
        
    def get_reward(self, action) -> float:
        """
        TODO: Returns reward based on state and action
        """
        return np.random.random()
    
    def reset(self) -> np.ndarray:
        """
        TODO: Reset the environment to default state and returns ndarray with same shape as obs
        """
        return torch.zeros(self.obs_space_shape)