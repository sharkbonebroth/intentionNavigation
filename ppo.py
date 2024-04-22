import torch
import torch.nn as nn
import numpy as np
from params import TrainingParameters, EnvParameters
import gym
from robot import Robot
from utilTypes import Action, Trajectory
from map import Map

class ReplayBuffer:
    def __init__(self, num_steps : int, num_envs : int, obs_space_shape : tuple, act_space_shape : tuple):
        batch_shape = (num_steps, num_envs)
        self.observations = torch.zeros(batch_shape + obs_space_shape).to(device)
        self.logprobs = torch.zeros(batch_shape).to(device)
        self.actions = torch.zeros(batch_shape + act_space_shape).to(device)
        self.advantages = torch.zeros(batch_shape).to(device)
        self.returns = torch.zeros(batch_shape).to(device)
        self.values = torch.zeros(batch_shape).to(device)
        
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
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values
        
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def normalize_fn(x : np.ndarray):
    return (x - x.mean()) / (x.std() + 1e-8)

def product(iter):
    val = 1
    for item in iter:
        val *= item
    return val

class ActorCritic(nn.Module):
    def __init__(self, obs_space_shape, action_space_shape):
        super().__init__()
        obs_space_shape = product(obs_space_shape)
        action_space_shape = product(action_space_shape)
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_space_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.)
        )
        
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_space_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_space_shape), std=0.01)
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_space_shape))

class PPO:
    def __init__(self, obs_space_shape, action_space_shape):
        self.policy = ActorCritic(obs_space_shape, action_space_shape).to(device)
        self.optimizer = torch.optim.Adam([
            {'params' : self.policy.actor_mean.parameters(), 'lr' : TrainingParameters.lr_actor},
            {'params' : self.policy.actor_logstd, 'lr' : TrainingParameters.lr_actor},
            {'params' : self.policy.critic.parameters(), 'lr' : TrainingParameters.lr_critic}
        ])
        
        self.obs_space_shape = product(obs_space_shape)
        self.action_space_shape = product(action_space_shape)
        
    def get_action_and_value(self, state : np.ndarray , action=None):
        state = torch.reshape(state, (-1, self.obs_space_shape))
        # logits are unnormalized action probs
        action_mean = self.policy.actor_mean(state)
        action_logstd = self.policy.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        value = self.get_value(state)
        return action , probs.log_prob(action).sum(1) , probs.entropy().sum(1), value
        
    def get_value(self, state):
        state = torch.reshape(state, (-1, self.obs_space_shape))
        value = self.policy.critic(state)
        return value
    
class IntentionNavEnv(gym.Env):
    MAX_STEPS = 10000
    def __init__(self, obs_space_shape : tuple, pathIn : Trajectory, mapIn : Map):
        self.done : bool = False
        self.obs_space_shape : tuple = obs_space_shape
        self.path : Trajectory = pathIn
        self.map : Map = mapIn
        self.robot = Robot(0.0, 0.0, 0.0)
        self.prevRobotPoseWorld = self.robot.getCurrentRobotPosWorld()
        self.intention = self.path.getIntention()
        self.steps = 0
        
    def step(self, action : np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        self.robot.move(Action(action)) #FROM LIYANG: NEED TO PASS IN MAP
        
        # Append intention to obs
        obs = self.robot.getFeedbackImage() #FROM LIYANG: NEED TO PASS IN MAP
        
        curRobotPoseWorld = self.robot.getCurrentRobotPosWorld()
        reward = self.get_reward(action, curRobotPoseWorld, self.prevRobotPoseWorld)
        
        done = self.is_done()
        self.prevRobotPoseWorld = curRobotPoseWorld
        
        info = dict()
        
        self.steps += 1
        return obs, reward, done, info
    
    def get_reward(self, action : np.ndarray, curRobotPos, prevRobotPos):
        # Add reward calculation for path following
        return 0.0
    
    def is_done(self):
        # Add goal check here
        
        if self.steps >= IntentionNavEnv.MAX_STEPS:
            return True
        return False
    
class DummyIntentionNavEnv(gym.Env):
    def __init__(self, obs_space_shape):
        self.done : bool = False
        self.obs_space_shape = obs_space_shape
        
    def step(self, action : np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
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
    
def get_env():
    return DummyIntentionNavEnv(EnvParameters.OBS_SPACE_SHAPE)

def rollout(env : gym.Env, buffer : ReplayBuffer):
    next_obs = torch.Tensor(env.reset()).to(device)
    next_done = torch.zeros(TrainingParameters.N_ENVS).to(device)
    
    for step in range(TrainingParameters.N_STEPS):
        buffer.observations[step] = next_obs
        buffer.dones[step] = next_done
        
        with torch.no_grad():
            action, logprob, _, value = ppo.get_action_and_value(next_obs)
            buffer.values[step] = value.flatten()
        buffer.actions[step] = action
        buffer.logprobs[step] = logprob
        
        # Gym part
        next_obs, reward, done, info = env.step(action.cpu().numpy())
        done = np.array([done])
        buffer.rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
    return next_obs, next_done
        
def get_device() -> torch.device:
    device = torch.device('cpu')
    print("Device set to : cpu")
    # set device to cpu or cuda
    if torch.cuda.is_available(): 
        device = torch.device('cuda:0') 
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    print("============================================================================================")
    return device
        
if __name__ == "__main__":
    device = get_device()
    
    # TODO (Nielsen): Log to wandb
    # TODO (Nielsen): Parallelize across multiple environments
    
    batch_size = TrainingParameters.N_STEPS * TrainingParameters.N_ENVS
    
    ppo = PPO(EnvParameters.OBS_SPACE_SHAPE, EnvParameters.ACT_SPACE_SHAPE)
    buffer = ReplayBuffer(TrainingParameters.N_STEPS, TrainingParameters.N_ENVS, EnvParameters.OBS_SPACE_SHAPE, EnvParameters.ACT_SPACE_SHAPE)
    env = get_env()
    
    num_updates = TrainingParameters.TOTAL_TIMESTEPS // batch_size
    target_kl = None
    global_step = 0
    for update in range(1, num_updates+1):
        global_step += TrainingParameters.N_ENVS
        
        if TrainingParameters.ANNEAL_LR:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * TrainingParameters.lr_actor
            ppo.optimizer.param_groups[0]["lr"] = lrnow
            
        #policy rollout
        next_obs, next_done = rollout(env, buffer)

        #bootstrap reward if not done
        with torch.no_grad():
            next_value = ppo.get_value(next_obs).reshape(1,-1)
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
    
        b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = buffer.flatten_batch()
    
        # Optimizing network
        batch_indices = np.arange(batch_size)
        clipfracs = []
        for epoch in range(TrainingParameters.N_EPOCHS):
            np.random.shuffle(batch_indices)
            for start in range(0, batch_size, TrainingParameters.MINIBATCH_SIZE):
                end = start + TrainingParameters.MINIBATCH_SIZE
                minibatch_indices = batch_indices[start:end]
                
                _, newlogprob, entropy, newvalue = ppo.get_action_and_value(
                    b_obs[minibatch_indices], b_actions[minibatch_indices]
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
                
        
                
    