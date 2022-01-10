import numpy as np
import pandas as pd
import random
import traci
import torch
import torch.optim as optim
from ElegantRL_learning.DQN.net import QNet, QNetDuel, QNetTwin, QNetTwinDuel
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward','mask'))  # 命名元组


class Agent():
    def __init__(self, env, state_dim, mid_dim, action_dim, replayBuffer, alpha=None, gamma=None, epsilon0=None,
                 learning_rate=None, k=None, ra=None):
        self.learning_rate = learning_rate if learning_rate is not None else 0.01
        self.gamma = gamma if gamma is not None else 0.9
        self.epsilon = epsilon0 if epsilon0 is not None else 0.8  # 有的需要 有的不需要
        self.alpha = alpha if alpha is not None else 0.8

        self.state_dim = state_dim
        self.mid_dim = mid_dim
        self.action_dim = action_dim
        self.soft_update_tau = 2**-8

        self.cri = QNet(self.state_dim, self.mid_dim, self.action_dim)
        self.cri_target = QNet(self.state_dim, self.mid_dim, self.action_dim)
        self.criterion = torch.nn.SmoothL1Loss()

        self.env = env  # 每个智能体有一个自己的环境对应  就是environment对象 然后从环境中获取环境的相关信息 状态巴拉巴拉
        self.replayBuffer = replayBuffer  # 对应经验缓冲池
        self.optimizer = optim.RMSprop(self.cri.parameters(), lr=self.learning_rate)
        self.k = 0.55 if k is None else k
        self.ra = 0.1 if ra is None else ra

    def get_state(self,*args):
        return self.env.get_state(*args)

    def select_action(self, state):
        # 选择动作  传入当前状态
        self.epsilon = max(self.epsilon*0.99,0.05)
        if np.random.rand() <= self.epsilon:
            # return random.randrange(self.action_dim,size=(state.shape[0]))
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                action = self.cri(state)
            # print(action)
            return action.argmax(dim=1)

    def take_action(self, current_action, step):
        now_step = self.env.step(current_action, step, 30)
        reward,delay_time = self.env.get_reward(now_step)
        waiting_time = self.env.get_total_waiting_time()
        queue = self.env.get_total_lanes_queue()
        return now_step, reward, waiting_time,delay_time,queue


    def optimize_model(self):
        if self.replayBuffer.get_size() < self.replayBuffer.minibatch:
            return
        transitions = self.replayBuffer.sample()
        # 拼接我们的transition们
        '''
        Transition([0,2],[2],[2,3],4)
        transitions : 
                [Transition(state=[0, 1], action=[1], next_state=[1, 2], reward=3), Transition(state=[0, 2], action=[2], next_state=[2, 3], reward=4)]
        batch : 
                Transition(state=([0, 1], [0, 2]), action=([1], [2]), next_state=([1, 2], [2, 3]), reward=(3, 4))
        '''
        batch = Transition(*zip(*transitions))

        q_label = []

        for i in range(len(batch.state)):
            state = torch.as_tensor(batch.state[i], dtype=torch.float)
            next_state = torch.as_tensor(batch.next_state[i], dtype=torch.float)
            mask = batch.mask[i]
            reward = batch.reward[i]

            state_value = self.cri(state).max(0)[0]

            next_state_value = self.cri_target(next_state).max(0)[0]

            ql = state_value + self.alpha * (reward + mask * next_state_value)
            q_label.append(ql)
        q_label = torch.cat([ql.unsqueeze(0) for ql in q_label], 0)
        # 计算损失
        q_list= []
        for state in batch.state:
            q = self.cri(state)
            q_list.append(q)

        q_list = torch.cat([q1 for q1 in q_list], 0)

        loss = self.criterion(q_list, q_label)
        # 优化模型
        self.optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def optimize_target_model(self):
        for tar,cur in zip(self.cri_target.parameters(),self.cri.parameters()):
            tar.data.copy_(cur.data*self.soft_update_tau + cur.data*(1-self.soft_update_tau))
