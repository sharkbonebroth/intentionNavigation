import torch
import numpy as np
from params import TrainingParameters, EnvParameters, NetParameters, WandbSettings
import gymnasium

from dataLoader import DataLoader
import wandb
import time
from net import product, ActorCritic
from env import IntentionNavEnv
from utilTypes import getIntentionAsOnehot
import os

STD_SCALE = 0.8
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
        
        self.best_weights = {}
        
        self.obs_space_shape = product(obs_space_shape)
        self.action_space_shape = product(action_space_shape)
        
    def get_action_and_value(self, obs, intention, action=None):
        obs = obs.to(self.device)
        onehot_intention = torch.from_numpy(getIntentionAsOnehot(intention, onehotSize=NetParameters.VECTOR_LEN)).to(self.device)
        action_mean, action_logstd, value = self.policy(obs, onehot_intention)
        action_std = STD_SCALE * torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action , probs.log_prob(action).sum(1) , probs.entropy().sum(1), value
        
    def get_value(self, obs, intention):
        obs = obs.to(self.device)
        onehot_intention = torch.from_numpy(getIntentionAsOnehot(intention, onehotSize=NetParameters.VECTOR_LEN)).to(self.device)
        _, _, value = self.policy(obs, onehot_intention)
        return value
    
    def get_state(self):
        return {
            'state_dict': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
    
    def save_model(self, state, filePath):
        torch.save(state, filePath)
        
    def load_model(self, filePath):
        state = torch.load(filePath)
        self.policy.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])

best_episode_reward = -float('inf')
best_weights = None
def rollout(env : gymnasium.Env, buffer : ReplayBuffer, global_step : int):
    global best_episode_reward, best_weights
    env.reset()
    obs, intention = env.getObservations()
    next_obs = torch.Tensor(obs).to(device)
    next_intention = torch.tensor(intention).to(device).view(-1)
    next_done = torch.zeros(TrainingParameters.N_ENVS).to(device)
    
    start = time.time()
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
        # env.render(next_obs)
        buffer.rewards[step] = torch.tensor(reward).to(device).view(-1)
        if done:
            env.reset()
            next_obs, next_intention = env.getObservations()
            next_done = False
        done = np.array([done])
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
        next_intention = torch.tensor(next_intention).to(device).view(-1)
        
        episode_reward = info["episode"]["reward"]
        
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            best_weights = ppo.get_state()
        
        print(f"global_step={global_step}, episodic_return={info['episode']['reward']}, episode_length={info['episode']['length']}")
        if WandbSettings.ON and ((step % WandbSettings.LOGGING_INTERVAL) == 0) and (step != 0):
            print("Logging episodic metrics")
            wandb.log({"charts/episodic_return" : episode_reward}, global_step)
            wandb.log({"charts/episodic_length" : info["episode"]["length"]}, global_step)

        if (step != 0) and ((global_step % NetParameters.SAVING_INTERVAL) == 0):
            print("Saving model...")
            ppo.save_model(ppo.get_state(), f"{NetParameters.MODEL_FOLDER}/latest.pth")
            ppo.save_model(best_weights, f"{NetParameters.MODEL_FOLDER}/best.pth")

    print(f"{time.time() - start} seconds for an ep")
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
    
    # Clockwise positive for yaw
    
    return IntentionNavEnv(NetParameters.FOV_SIZE, dataLoader)

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

    model_directory = "./models"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    # TODO (Nielsen): Parallelize across multiple environments
    batch_size = TrainingParameters.N_STEPS * TrainingParameters.N_ENVS
    ppo = PPO(device, EnvParameters.OBS_SPACE_SHAPE, EnvParameters.ACT_SPACE_SHAPE)
    if NetParameters.LOAD_MODEL:
        ppo.load_model(NetParameters.MODEL_PATH)
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
                
        
                
    