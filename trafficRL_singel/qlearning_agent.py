import numpy as np
import pandas as pd
import random
import traci
from ElegantRL_learning.DQN.net import QNet, QNetDuel, QNetTwin, QNetTwinDuel


class Agent():
    def __init__(self, learning_rate, env, action_dim, state_dim, mid_dim):
        self.learning_rate = learning_rate
        self.gamma = 0.9
        self.epsilon = 0  # 有的需要 有的不需要
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.mid_dim = mid_dim
        self.net = QNetTwinDuel(self.mid_dim, self.state_dim, self.action_dim)

        self.env = env  # 每个智能体有一个自己的环境对应  就是environment对象 然后从环境中获取环境的相关信息 状态巴拉巴拉
        # self.replayBuffer = replayBuffer  # 对应经验缓冲池
        self._q_table = self._init_q_table()

    def _init_q_table(self):
        #
        '''
        function：创建以及初始化q表
        structure: 三维字典吧
        [[[等待队列的长度列表],[p0],[p1],[p2],[p3],[p4],[p5],[p6],[p7]],
        [[S0],0,0,...,],
        ...,
        ]
        考虑是否将等待队列长度st转换成 长度排序的[]
        :return:
        '''

        _q_table = pd.DataFrame(columns=['s',0,1,2,3,4,5,6,7])

        return _q_table

    def update(self,step,state,a,new_a):
        # 更新q表
        reward = self.env.get_reward(step)
        new_state = self.get_state()
        q = self._q_table
        q[state][a] = q[state][a] + self.alpha *(reward +self.gamma*q[new_state][new_a]-q[state][a])

        # 冷启动  选择了这个动作 才会去更新其q值

    def get_state(self):
        return self.env.get_state()

    def choose_action(self, state):
        # 选择动作  传入当前状态
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)

        Action = model.predict(state)
        return np.argmax(action[0])

    def take_action(self, current_action, step):
        step = self.env.step(current_action, step)
        return step



    def explore(self):
        '''
        在环境中探索，返回一个trajectory(轨迹)[st,at,rt,st+1]
        将其存入memorybuffer
        :return:
        '''
        pass

    # def train(self):
    #     traj_list = self.replayBuffer.sample_batch()