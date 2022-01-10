import numpy as np
import pandas as pd
import random
import traci
import torch
import torch.optim as optim
from ElegantRL_learning.DQN.net import QNet, QNetDuel, QNetTwin, QNetTwinDuel
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))  # 命名元组


class Agent():
    def __init__(self, env, state_dim, mid_dim, action_dim, replayBuffer, alpha=None, gamma=None, epsilon0=None,
                 learning_rate=None, k=None, ra=None):
        self.learning_rate = learning_rate if learning_rate is not None else 0.9
        self.gamma = gamma if gamma is not None else 0.9
        self.epsilon = epsilon0 if epsilon0 is not None else 0.8  # 有的需要 有的不需要
        self.alpha = alpha if alpha is not None else 0.8

        self.state_dim = state_dim
        self.mid_dim = mid_dim
        self.action_dim = action_dim
        self.soft_update_tau = 2**-8

        self.cri = QNet(self.state_dim, self.mid_dim, self.action_dim)
        # # print(self.net)
        self.cri_target = QNet(self.state_dim, self.mid_dim, self.action_dim)

        # print(self.net)


        self.env = env  # 每个智能体有一个自己的环境对应  就是environment对象 然后从环境中获取环境的相关信息 状态巴拉巴拉
        self.replayBuffer = replayBuffer  # 对应经验缓冲池
        self.optimizer = optim.RMSprop(self.cri.parameters(), lr=self.learning_rate)
        self.k = 0.55 if k is None else k
        self.ra = 0.1 if ra is None else ra

    # def update(self,step,state,a,new_a):
    #     # 更新q
    #     self.net.forward(state)
    #     reward = self.env.get_reward(step)
    #     new_state = self.get_state()
    #     q = self.
    #     q[state][a] = q[state][a] + self.alpha *(reward +self.gamma*q[new_state][new_a]-q[state][a])
    #
    #     # 冷启动  选择了这个动作 才会去更新其q值

    def get_state(self):
        return self.env.get_state()

    def select_action(self, state):
        # 选择动作  传入当前状态
        if np.random.rand() <= self.epsilon:
            # return random.randrange(self.action_dim,size=(state.shape[0]))
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                action = self.cri(state)
            # print(action)
            return action.argmax(dim=1)

    def take_action(self, current_action, step, flowTag, flowIndex):
        # 2、此处增加判断标志和current_action是否相符
        '''
            传入action，n_step   要执行的动作 和要执行多少步
            然后判断是否切换相位
            如果切换相位先执行last_action+1相位的3step
            判断action
            如果是0，1，2，3 自动*2 然后执行n_step步
            如果是4，5 执行n_step=12步
        '''
        if flowTag != 'ba':
            if flowTag == 'k':
                if flowIndex == current_action:
                    n_step = 15
                    now_step = self.env.step(flowIndex, step, n_step)
                else:
                    n_step = 12
                    now_step = self.env.step(flowIndex, step, n_step)
                n_step = 30
                now_step = self.env.step(current_action, now_step, n_step)  # 判断没有切换相位 执行30步
            else:
                n_step = 12
                if flowIndex == 0 or flowIndex == 2:
                    action = 4
                    now_step = self.env.step(action, step, n_step)
                else:
                    action = 5
                    now_step = self.env.step(action, step, n_step)
                n_step = 30
                now_step = self.env.step(current_action, now_step, n_step)  # 判断切换相位了，自动执行3步 + n_step步
            reward = self.env.get_reward(now_step)
        else:
            now_step = self.env.step(current_action, step, 30)
            reward = self.env.get_reward(now_step)

        return now_step, reward

    def identifyFlow(self):
        '''

        :return: k and ra
        首先计算四个k
        再计算ra
        '''
        k_list = []

        q_list = []
        for edge in self.env.inEdges:
            q_list.append(self.env.get_edge_queue(edge))

        if sum(q_list) == 0:
            k = 0
        else:
            for q in q_list:
                k_list.append(round(q/sum(q_list), 2))
            k = max(k_list)

        ra_list = []
        ls, ss, rs = self.env.get_turn_number_inEdges()

        for i in range(len(ls)):
            if (ls[i] + ss[i] + rs[i]) != 0:
                Rs = round((ss[i] + rs[i]) / (ls[i] + ss[i] + rs[i]), 2)
                Rl = round(ls[i] / (ls[i] + ss[i] + rs[i]), 2)
                ra_list.append(Rl - Rs)
            else:
                ra_list.append(0)
        ra = max(ra_list)
        if k >= self.k:
            return ["k", k_list.index(k)]
        elif ra >= self.ra:
            return ["ra", ra_list.index(ra)]
        else:
            return None

    def explore(self):
        '''
        在环境中探索，返回一个trajectory(轨迹)[st,at,rt,st+1]
        将其存入memorybuffer
        :return:
        '''
        pass

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
        # print(batch)
        # print("len batch.state: ",len(batch.state),len(batch.state[0]))
        # print("len batch.action: ", len(batch.action))
        # print("len batch.reward: ", len(batch.reward))
        state_tensor_list = [torch.as_tensor(i) for i in batch.state]
        state_batch = torch.cat(state_tensor_list)

        action_batch = torch.tensor(batch.action)
        # print("action batch:",len(action_batch))
        reward_batch = torch.tensor(batch.reward)

        # res = self.net(state_batch)
        # print(res.shape())

        state_action_values = []
        for state in batch.state:
            state_action_values.append(self.cri(torch.as_tensor(state, dtype=torch.float)))

        state_action_values = torch.cat([s for s in state_action_values], 0)
        state_action_values = state_action_values.max(1)[0].detach()
        # print("state_action_values:", state_action_values)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)
        # non_final_mask = tensor([True, True])
        l = [torch.tensor(s, dtype=torch.float) for s in batch.next_state if s is not None]
        non_final_next_states = []
        for next_state in l:
            non_final_next_states.append(self.cri_target(next_state))

        non_final_next_states = torch.cat([s.unsqueeze(0) for s in non_final_next_states], 0)
        # print("non_final_next_states.type:", non_final_next_states.type())
        # print("non_final_mask:", non_final_mask)
        next_state_values = torch.zeros(self.replayBuffer.minibatch, dtype=torch.float)
        # print("next_state_values type",next_state_values.type())
        # print("non_final_next_states.max(1)[1] type ",non_final_next_states.max(1)[1].type())
        next_state_values[non_final_mask] = non_final_next_states.max(1)[0].detach()
        # 行维度  .max(1)[0] 返回values的最大值列表 .max(1)[1]返回最大值index列表
        # 列维度 .max(0)[0] 返回values的最大值列表 .max(0)[1]返回最大值index列表
        # print("next_state_values:",next_state_values)

        # expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # print("state_action_values length:",len(state_action_values))
        # print("next_state_values length:",len(next_state_values))
        # print("reward_batch length",len(reward_batch)) # 100
        # print("reward_batch+self.gamma*next_state_values-state_action_values:",reward_batch+self.gamma*next_state_values-state_action_values)
        expected_state_action_values = state_action_values + self.alpha * (
                reward_batch + self.gamma * next_state_values - state_action_values)
        # print("expected_state_action_values : ",expected_state_action_values)
        # print(expected_state_action_values.unsqueeze(1))
        # print(state_action_values.unsqueeze(1))
        # 计算损失
        loss_func = torch.nn.SmoothL1Loss()
        loss = loss_func(state_action_values.unsqueeze(1), expected_state_action_values.unsqueeze(1))

        # 优化模型
        self.optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()

    def optimize_target_model(self):
        for tar,cur in zip(self.cri_target.parameters(),self.cri.parameters()):
            tar.data.copy_(cur.data*self.soft_update_tau + cur.data*(1-self.soft_update_tau))
