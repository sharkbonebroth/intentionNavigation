from robot import Robot
from utilTypes import Action, find_closest_point, get_distance
from map import Map, MAPSCALE
from typing import Tuple
import matplotlib.pyplot as plt
import gymnasium
import numpy as np
import torch
from skimage.draw import disk, line
from skimage.transform import resize
from skimage import util
from dataLoader import DataLoader, TrainingDataPoint
import enum

class DoneState(int, enum.Enum):
    NOT_DONE = 0
    CRASH = 1
    GOAL = 2
    MAX_STEPS = 3

class Reward:
    CRASHING : float = -5.0
    ININFLATIONZONE : float = -2.5
    REGRESSPOINTPENALTY: float = -0.2
    MAXPROGRESSPOINTREWARD: float = 0.2
    STAGNATEPENALTY: float = -0.05
    VELOCITYTOOLOWPENALTY: float = -0.02
    VELOCITYTOOLOWTHRESHOLD: float = 1.0
    GOAL : float = 5.0
    ACTION : float = -0.3

MAX_ANG_VELOCITY = np.pi
MAX_LIN_VELOCITY = 3.0
MIN_LIN_VELOCITY = -1.0

class IntentionNavEnv(gymnasium.Env):
    MAX_STEPS = 10000
    def __init__(self, obs_space_shape : Tuple, dataLoader: DataLoader):
        self.action_space = gymnasium.spaces.Box(np.array([MIN_LIN_VELOCITY,-MAX_ANG_VELOCITY]), np.array([MAX_LIN_VELOCITY,MAX_ANG_VELOCITY])) 
        
        self.dataLoader = dataLoader
        currLabelledData, map = self.dataLoader.getLabelledDataAndMap()
        self.setLabelledDataAndMap(currLabelledData, map)
        
        self.done : bool = False
        self.obs_space_shape : tuple = obs_space_shape
    
        self.trainingId = 0
        self.prevClosestDistance = 100000

        plt.figure(figsize=(6, 6), dpi=200)

    def setLabelledDataAndMap(self, labelledData: TrainingDataPoint, map: Map):
        self.curPath = labelledData.trajectory
        self.curIntention = labelledData.direction
        map.registerCorrectTrajOnMap(self.curPath)
        self.robot = Robot(map=map, startX=labelledData.startPoint[0], startY=labelledData.startPoint[1], yaw=labelledData.startPoint[2], numOdomToPlot=200)
        self.startPoint = labelledData.startPoint
        self.endPoint = labelledData.endPoint
        self.prevRobotPoseWorld = self.robot.currPositionActual
        self.steps = 0
        self.prevWaypointID = 0
        self.curBestWaypointId = 0
        self.totalReward = 0

        
    def getObservations(self):
        return self.robot.getBinaryFeedbackImage(scaleFactor=4), float(self.curIntention)
        
    def step(self, action : np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        self.robot.move(Action(*action)) 
        
        obs, intention = self.getObservations()
        
        curRobotPoseWorld = self.robot.getRobotPoseWorld()
        doneState = self.is_done(curRobotPoseWorld)
        reward = self.get_reward(action, curRobotPoseWorld, doneState)
        
        # Reward clipping
        # reward = np.clip(reward, -10, 10)
        
        done = doneState != DoneState.NOT_DONE
            
        self.prevRobotPoseWorld = curRobotPoseWorld
        
        info = dict()
        self.totalReward += reward
        self.steps += 1
        info['episode'] = {
            'reward' : self.totalReward / self.steps,
            'length' : self.steps
        }
        return obs, intention, reward, done, info
    
    def get_reward(self, action : np.ndarray, curRobotPos, doneState):
        # closestWaypointId = find_closest_point(curRobotPos[:2], self.curPath)

        # if closestWaypointId >= self.prevWaypointID:
        #     # Reward proportional to how close it is to the closest waypoint
        #     reward = Reward.MAXPROGRESSPOINTREWARD / (1 + get_distance(self.robot.currPositionActual, self.curPath[closestWaypointId]))
            
        #     # Penalize if it stagnates
        #     if closestWaypointId == self.prevWaypointID:
        #         reward += Reward.STAGNATEPENALTY
        # else:
        #     # Penalize!
        #     reward = Reward.REGRESSPOINTPENALTY - get_distance(self.robot.currPositionActual, self.curPath[self.curBestWaypointId])
        
        # if self.robot.hasCrashedIntoWall():
        #     reward += Reward.CRASHING
        # elif self.robot.isInInflationZone():
        #     reward += Reward.ININFLATIONZONE

        # if action[0] < Reward.VELOCITYTOOLOWTHRESHOLD:
        #     reward += Reward.VELOCITYTOOLOWPENALTY
        
        # if closestWaypointId > self.curBestWaypointId:
        #     self.curBestWaypointId = closestWaypointId
        
        # self.prevWaypointID = closestWaypointId

        # return reward
        
        if doneState == DoneState.GOAL:
            return Reward.GOAL
        elif self.robot.hasCrashedIntoWall():
            return Reward.CRASHING
        closestWaypointId = find_closest_point(curRobotPos[:2], self.curPath)
        
        reward = Reward.ACTION
        
        inverse = 1
        if closestWaypointId < self.curBestWaypointId:
            inverse *= -1
        distToTarget = get_distance(curRobotPos, self.curPath[self.curBestWaypointId])
        reward = inverse * distToTarget
        reward += abs(action[0]) * 0.1
        
        # TODO: get angle2goal cost
        
        # reward = 1 / (1+distToTarget)
        
        if self.robot.isInInflationZone():
            reward += Reward.ININFLATIONZONE
        
        if closestWaypointId > self.curBestWaypointId:
            self.curBestWaypointId = closestWaypointId
        return reward
    
    def reset(self):
        # currLabelledData, map = self.dataLoader.getLabelledDataAndMap()
        currLabelledData, map = self.dataLoader.getLabelledDataAndMapAtId(0)
        self.setLabelledDataAndMap(currLabelledData, map)
        # if self.trainingId >= len(self.paths):
        #     return np.zeros((640,480))
        # self.trainingId +=1
        # self.curPath = self.paths[self.trainingId]
        # self.curIntention = self.intentions[self.trainingId]
        
    
    def is_done(self, curRobotPoseWorld) -> DoneState:
        if self.robot.hasCrashedIntoWall():
            print("Crashed!!")
            return DoneState.CRASH
        if abs(len(self.curPath) - self.curBestWaypointId) < 5:
            print("GOALLLLL!!!!")
            return DoneState.GOAL
        if get_distance(curRobotPoseWorld, self.endPoint) < 0.1:
            print("GOALLLLL!!!!")
            return DoneState.GOAL
        if self.steps >= IntentionNavEnv.MAX_STEPS:
            return DoneState.MAX_STEPS
        return DoneState.NOT_DONE
    
    
    def render(self, feedbackImage: np.ndarray):
        img = np.copy(self.robot.mapImgWithTrajAndPerfectOdomPlotted)
        
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

        # Get the odom image
        feedbackImage = self.robot.convertBinaryFeedbackImageToColor(feedbackImage)
        feedbackImageResized = resize(feedbackImage, (img.shape[0], img.shape[1]), anti_aliasing=True)
        feedbackImageResized = util.img_as_ubyte(feedbackImageResized)

        # Final Image
        finalImg = np.concatenate([img, feedbackImageResized], axis=0)

        plt.clf()
        plt.imshow(finalImg)
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