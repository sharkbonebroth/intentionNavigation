from params import EnvParameters, NetParameters
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from torch.cuda.amp.autocast_mode import autocast
from einops import rearrange

def product(iter):
    val = 1
    for item in iter:
        val *= item
    return val

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, obs_space_shape, action_space_shape):
        super().__init__()
        obs_space_shape = product(obs_space_shape)
        action_space_shape = product(action_space_shape)
        
        self.num_channel = NetParameters.NUM_CHANNEL
        
        # observation encoder
        self.conv1 = nn.Conv2d(self.num_channel, NetParameters.NET_SIZE // 4, 3, 1, 1)
        self.conv1a = nn.Conv2d(NetParameters.NET_SIZE // 4, NetParameters.NET_SIZE // 4, 3, 1, 1)
        self.conv1b = nn.Conv2d(NetParameters.NET_SIZE // 4, NetParameters.NET_SIZE // 4, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(NetParameters.NET_SIZE // 4, NetParameters.NET_SIZE // 2, 2, 1, 1)
        self.conv2a = nn.Conv2d(NetParameters.NET_SIZE // 2, NetParameters.NET_SIZE // 2, 2, 1, 1)
        self.conv2b = nn.Conv2d(NetParameters.NET_SIZE // 2, NetParameters.NET_SIZE // 2, 2, 1, 1)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(NetParameters.NET_SIZE // 2, NetParameters.NET_SIZE - NetParameters.INTENTION_SIZE, 3,
                               1, 0)
        self.conv3a = nn.Conv2d(NetParameters.NET_SIZE - NetParameters.INTENTION_SIZE, NetParameters.NET_SIZE - NetParameters.INTENTION_SIZE, 3,
                               1, 0)
        self.conv3b = nn.Conv2d(NetParameters.NET_SIZE - NetParameters.INTENTION_SIZE, NetParameters.NET_SIZE - NetParameters.INTENTION_SIZE, 3,
                               1, 0)
        self.pool3 = nn.MaxPool2d(2)
        
        self.fully_connected_1 = nn.Linear(NetParameters.VECTOR_LEN, NetParameters.INTENTION_SIZE)
        self.fully_connected_2 = nn.Linear(NetParameters.NET_SIZE, NetParameters.NET_SIZE)
        self.fully_connected_3 = nn.Linear(NetParameters.NET_SIZE, NetParameters.NET_SIZE)
        
        self.policy_layer = layer_init(nn.Linear(NetParameters.NET_SIZE, action_space_shape), std=0.01)
        self.value_layer = layer_init(nn.Linear(NetParameters.NET_SIZE, 1), std=1.)
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_space_shape))
        
    @autocast()
    def forward(self, obs, intention):
        """run neural network"""
        obs = torch.reshape(obs, (-1, self.num_channel, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE))
        intention = torch.reshape(intention, (-1, NetParameters.VECTOR_LEN))
        
        print(obs.shape)
        
        # matrix input
        x_1 = F.relu(self.conv1(obs))
        x_1 = F.relu(self.conv1a(x_1))
        x_1 = F.relu(self.conv1b(x_1))
        x_1 = self.pool1(x_1)
        
        print(x_1.shape)
        
        x_1 = F.relu(self.conv2(x_1))
        x_1 = F.relu(self.conv2a(x_1))
        x_1 = F.relu(self.conv2b(x_1))
        x_1 = self.pool2(x_1)
        
        print(x_1.shape)
        
        x_1 = self.conv3(x_1)
        x_1 = self.conv3a(x_1)
        x_1 = self.conv3b(x_1)
        x_1 = self.pool3(x_1)
        
        print(x_1.shape)
        
        x_1 = F.relu(x_1.view(x_1.size(0), -1))
        
        # vector input
        x_2 = F.relu(self.fully_connected_1(intention))
        
        x_3 = torch.cat((x_1, x_2), -1)
        h1 = F.relu(self.fully_connected_2(x_3))
        h1 = self.fully_connected_3(h1)
        h2 = F.relu(h1 + x_3)
        h2 = h2.view(h2.shape[0], h2.shape[1], 1, 1)
        
        x = rearrange(h2, 'b c h w -> b (h w) c')
        
        x = torch.reshape(x, (-1, NetParameters.NET_SIZE))
        
        actor_mean = self.policy_layer(x)
        actor_logstd = self.actor_logstd.expand_as(actor_mean)
        value = self.value_layer(x)
        return actor_mean, actor_logstd, value