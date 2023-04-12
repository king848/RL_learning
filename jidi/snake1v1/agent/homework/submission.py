import copy

import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
# ====================================== helper functions ======================================
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from common import *


# ====================================== define algo ===========================================
# todo
class Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear3(x)
        return x


# todo
class DQN(object):
    def __init__(self):
        # pass
        self.state_dim = 18
        self.action_dim = 4
        self.hidden_size = 256

        self.critic_eval = Critic(self.state_dim, self.action_dim, self.hidden_size)
    def choose_action(self, observation):
        # my_action
        obs = get_observations(observation, 0, 18)
        obs = torch.tensor(obs, dtype=torch.float).view(1, -1)
        my_action = torch.argmax(self.critic_eval(obs)).item()
        #enemy_action
        observation1=copy.deepcopy(observation)
        observation1['controlled_snake_index']=5-observation['controlled_snake_index']
        obs1 = get_observations(observation1, 0, 18)
        obs1 = torch.tensor(obs1, dtype=torch.float).view(1, -1)
        enemy_action = torch.argmax(self.critic_eval(obs1)).item()
        #判断下一步是否会撞上,若会撞上，更新动作
        my_action = check_action(observation,my_action,enemy_action)

        return my_action

    def load(self, file):
        # pass
        base_path = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(base_path, file)
        self.critic_eval.load_state_dict(torch.load(file))


def to_joint_action(actions, num_agent):
    joint_action = []
    for i in range(num_agent):
        action = actions
        one_hot_action = [0] * 4
        one_hot_action[action] = 1
        joint_action.append(one_hot_action)
    return joint_action


# ===================================== define agent =============================================
#todo
agent = DQN()
agent.load('critic_20000.pth')


# ================================================================================================
"""
input:
    observation: dict
    {
        1: 豆子，
        2: 第一条蛇的位置，
        3：第二条蛇的位置，
        "board_width": 地图的宽，
        "board_height"：地图的高，
        "last_direction"：上一步各个蛇的方向，
        "controlled_snake_index"：当前你控制的蛇的序号（2或3）
        }
return: 
    action: eg. [[[0,0,0,1]]]
"""
# todo
def my_controller(observation, action_space_list, is_act_continuous):
    # pass
    action = agent.choose_action(observation)
    action_ = to_joint_action(action,1)

    # [0,0,0,1]
    return action_
