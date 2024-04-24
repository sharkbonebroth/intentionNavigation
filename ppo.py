import torch
import numpy as np
from params import TrainingParameters, EnvParameters, NetParameters
import gymnasium
from robot import Robot
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F
from einops import rearrange
from skimage.draw import disk, line
from utilTypes import Action, trajectoryType, find_closest_point, Intention
from map import Map, MAPSCALE
from typing import Tuple, List
from dataLoader import DataLoader
import wandb
import time
import gymnasium
from env import IntentionNavEnv
import math
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self, num_steps : int, num_envs : int, obs_space_shape : tuple, act_space_shape : tuple):
        batch_shape = (num_steps, num_envs)
        self.observations = torch.zeros(batch_shape + obs_space_shape).to(device)
        self.logprobs = torch.zeros(batch_shape).to(device)
        self.actions = torch.zeros(batch_shape + act_space_shape).to(device)
        self.advantages = torch.zeros(batch_shape).to(device)
        self.returns = torch.zeros(batch_shape).to(device)
        self.values = torch.zeros(batch_shape).to(device)
        self.intentions = torch.zeros(batch_shape).to(device)
        
        self.rewards = torch.zeros(batch_shape).to(device)
        self.dones = torch.zeros(batch_shape).to(device)
        
        self.obs_space_shape = obs_space_shape
        self.act_space_shape = act_space_shape
        
    def flatten_batch(self):
        b_obs = self.observations.reshape((-1, ) + self.obs_space_shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1, ) + self.act_space_shape)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)
        b_intentions = self.intentions.reshape(-1)
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_intentions

def normalize_fn(x : np.ndarray):
    return (x - x.mean()) / (x.std() + 1e-8)

class PPO:
    def __init__(self, device, obs_space_shape, action_space_shape):
        self.device = device
        self.policy = ActorCritic(obs_space_shape, action_space_shape).to(self.device)
        total = 0
        id = 0
        for p in self.policy.parameters():
            total += p.numel()
            id += 1
        print("Total # of params ", total)
        self.optimizer = torch.optim.Adam([
            {'params' : self.policy.parameters(), 'lr' : TrainingParameters.lr_actor}
        ])
        
        self.obs_space_shape = product(obs_space_shape)
        self.action_space_shape = product(action_space_shape)
        
    def get_action_and_value(self, obs, intention, action=None):
        obs = obs.to(self.device)
        onehot_intention = torch.from_numpy(getIntentionAsOnehot(intention, onehotSize=NetParameters.VECTOR_LEN)).to(self.device)
        action_mean, action_logstd, value = self.policy(obs, onehot_intention)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action , probs.log_prob(action).sum(1) , probs.entropy().sum(1), value
        
    def get_value(self, obs, intention):
        obs = obs.to(self.device)
        onehot_intention = torch.from_numpy(getIntentionAsOnehot(intention, onehotSize=NetParameters.VECTOR_LEN)).to(self.device)
        _, _, value = self.policy(obs, onehot_intention)
        return value
    
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
        
        self.curPath = []
        self.curIntention = Intention.LEFT
        
    def getObservations(self):
        return self.robot.getFeedbackImage(), float(self.curIntention)

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
        endX = math.cos(robotYaw) * 10
        endY = math.sin(robotYaw) * 10
        rr, cc = line(robotImgY, robotImgX, endY, endX)
        img[rr, cc] = np.array([0, 255, 0])

        plt.imshow(img)
        plt.show()

        
    def step(self, action : np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self.robot.move(Action(*action)) 
        
        # Append intention to obs
        obs, intention = self.getObservations()
        
        curRobotPoseWorld = (0.0, 0.0, 0.0)
        reward = self.get_reward(action, curRobotPoseWorld, self.prevRobotPoseWorld)
        
        done = self.is_done()
        self.prevRobotPoseWorld = curRobotPoseWorld
        
        info = dict()
        
        self.steps += 1
        return obs, intention, reward, done, info
    
    def get_reward(self, action : np.ndarray, curRobotPos, prevRobotPos):
        
        # Add reward calculation for path following
        return 0.0
    
    def reset(self):
        if self.trainingId >= len(self.paths):
            return np.zeros((640,480))
        self.curPath = self.paths[self.trainingId]
        self.curIntention = self.intentions[self.trainingId]
        self.trainingId +=1
    
    def is_done(self):
        # Add goal check here
        
        if self.steps >= IntentionNavEnv.MAX_STEPS:
            return True
        return False
    
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

def rollout(env : gymnasium.Env, buffer : ReplayBuffer):
    env.reset()
    obs, intention = env.getObservations()
    next_obs = torch.Tensor(obs).to(device)
    next_intention = torch.tensor(intention).to(device).view(-1)
    next_done = torch.zeros(TrainingParameters.N_ENVS).to(device)
    
    for step in range(TrainingParameters.N_STEPS):
        global_step += 1 * TrainingParameters.N_ENVS
        buffer.observations[step] = next_obs
        buffer.intentions[step] = next_intention
        buffer.dones[step] = next_done
        
        with torch.no_grad():
            action, logprob, _, value = ppo.get_action_and_value(next_obs, next_intention)
            buffer.values[step] = value.flatten()
        buffer.actions[step] = action
        buffer.logprobs[step] = logprob
        
        # Gym part
        next_obs, next_intention, reward, done, info = env.step(action.cpu().numpy().flatten())
        done = np.array([done])
        buffer.rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
        next_intention = torch.tensor(next_intention).to(device).view(-1)
        
        print(f"global_step={global_step}, episodic_return={info['episode']['reward']}")
        if WandbSettings.ON:
            wandb.log({"charts/episodic_return" : info["episode"]["reward"]}, global_step)
            wandb.log({"charts/episodic_length" : info["episode"]["length"]}, global_step)
    return next_obs, next_intention, next_done, global_step
        
def get_device() -> torch.device:
    if torch.cuda.is_available(): 
        device = torch.device('cuda:0') 
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        device = torch.device('cpu')
        print("Device set to : cpu")
    print("============================================================================================")
    return device
        
def get_env():
    # return DummyIntentionNavEnv(EnvParameters.OBS_SPACE_SHAPE)
    dataLoader = DataLoader("maps", "labelledData") # mapdir, labelledDataDir
    paths = []
    intentions = []
    trainingData, map = dataLoader.getLabelledDataAndMap()
    trajInM = trainingData.trajectory
    paths.append(trajInM)
    intentions.append(trainingData.direction)
    
    # Clockwise positive for yaw
    startPoint = trainingData.startPoint
    endPoint = trainingData.endPoint
    
    robotMap = map
    
    return IntentionNavEnv(NetParameters.FOV_SIZE, pathsIn=paths, intentionsIn=intentions, mapIn=robotMap, startPoint=startPoint, endPoint=endPoint)

if __name__ == "__main__":
    device = get_device()
    
    if WandbSettings.ON:
        wandb_id = wandb.util.generate_id()
        wandb.init(project=WandbSettings.EXPERIMENT_PROJECT,
                    name=WandbSettings.EXPERIMENT_NAME,
                    id=wandb_id,
                    resume='allow')
        print('id is:{}'.format(wandb_id))
        print('Launching wandb...\n')

    # TODO (Nielsen): Parallelize across multiple environments
    
    batch_size = TrainingParameters.N_STEPS * TrainingParameters.N_ENVS
    
    ppo = PPO(device, EnvParameters.OBS_SPACE_SHAPE, EnvParameters.ACT_SPACE_SHAPE)
    buffer = ReplayBuffer(TrainingParameters.N_STEPS, TrainingParameters.N_ENVS, EnvParameters.OBS_SPACE_SHAPE, EnvParameters.ACT_SPACE_SHAPE)
    env = get_env()
    
    num_updates = TrainingParameters.TOTAL_TIMESTEPS // batch_size
    target_kl = None
    global_step = 0
    start_time = time.time()
    for update in range(1, num_updates+1):
        global_step += TrainingParameters.N_ENVS
        
        if TrainingParameters.ANNEAL_LR:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * TrainingParameters.lr_actor
            ppo.optimizer.param_groups[0]["lr"] = lrnow
            
        #policy rollout
        next_obs, next_intention, next_done, global_step = rollout(env, buffer, global_step)

        #bootstrap reward if not done
        with torch.no_grad():
            next_value = ppo.get_value(next_obs, next_intention).reshape(1,-1)
            advantages = torch.zeros_like(buffer.rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(TrainingParameters.N_STEPS)):
                if t == TrainingParameters.N_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - buffer.dones[t+1]
                    nextvalues = buffer.values[t+1]
                delta = buffer.rewards[t] + TrainingParameters.GAMMA * nextvalues * nextnonterminal - buffer.values[t]
                buffer.advantages[t] = lastgaelam = delta + TrainingParameters.GAMMA * TrainingParameters.LAM * nextnonterminal * lastgaelam
            returns = buffer.advantages + buffer.values
    
        b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_intentions = buffer.flatten_batch()
    
        # Optimizing network
        batch_indices = np.arange(batch_size)
        clipfracs = []
        for epoch in range(TrainingParameters.N_EPOCHS):
            np.random.shuffle(batch_indices)
            for start in range(0, batch_size, TrainingParameters.MINIBATCH_SIZE):
                end = start + TrainingParameters.MINIBATCH_SIZE
                minibatch_indices = batch_indices[start:end]
                
                _, newlogprob, entropy, newvalue = ppo.get_action_and_value(
                    b_obs[minibatch_indices], b_intentions[minibatch_indices], b_actions[minibatch_indices]
                )
                logratio = newlogprob - b_logprobs[minibatch_indices]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > TrainingParameters.EPS_CLIP).float().mean().item()]
                
                minibatch_advantages = b_advantages[minibatch_indices]
                minibatch_advantages = normalize_fn(minibatch_advantages)
                
                # Policy loss
                pg_loss1 = -minibatch_advantages * ratio
                pg_loss2 = -minibatch_advantages * torch.clamp(ratio, 1 - TrainingParameters.EPS_CLIP, 1 + TrainingParameters.EPS_CLIP)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                newvalue = newvalue.view(-1)
                value_loss_unclipped = torch.square(newvalue - b_returns[minibatch_indices])
                
                value_clipped = b_values[minibatch_indices] + torch.clamp(
                    newvalue - b_values[minibatch_indices],
                    -TrainingParameters.EPS_CLIP,
                    TrainingParameters.EPS_CLIP
                )
                value_loss_clipped = torch.square(value_clipped - b_returns[minibatch_indices])
                
                value_loss = torch.mean(torch.maximum(value_loss_unclipped, value_loss_clipped))
                
                entropy_loss = entropy.mean()
                
                loss = pg_loss - TrainingParameters.ENTROPY_COEF * entropy_loss + value_loss * TrainingParameters.VALUE_COEF
                
                ppo.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ppo.policy.parameters(), TrainingParameters.MAX_GRAD_NORM)
                ppo.optimizer.step()
            if target_kl is not None and target_kl > 0.15:
                break
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        if WandbSettings.ON:
            print("Logging here")
            wandb.log({"charts/learning_rate" : ppo.optimizer.param_groups[0]["lr"]}, global_step)
            wandb.log({"losses/value_loss" : value_loss.item()}, global_step)
            wandb.log({"losses/policy_loss" : pg_loss.item()}, global_step)
            wandb.log({"losses/entropy" : entropy_loss.item()}, global_step)
            wandb.log({"losses/old_approx_kl" : old_approx_kl.item()}, global_step)
            wandb.log({"losses/approx_kl" : approx_kl.item()}, global_step)
            wandb.log({"losses/clipfrac" : np.mean(clipfracs)}, global_step)
            wandb.log({"losses/explained_variance" : explained_var}, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            wandb.log({"charts/SPS" : int(global_step / (time.time() - start_time))}, global_step)
    
    if WandbSettings.ON:
        wandb.finish()
                
        
                
    