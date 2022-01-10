'''
存放不同的DRL算法
继承层级关系
DQN的Agent类
class AgentDQN
class AgentDuelingDQN  采用不同Q值计算方式
class AgentDoubleDQN  包含Q网络
class AgentD3QN  前两个结合

DDPG中的Agent类
class AgentBase
class AgentDDPG
class AgentTD3
class AgentSAC
class AgentPPO
class AgentModSAC
class AgentInterSAC
class AgentInterPPO等等
'''
import os
import torch
import numpy as np
import numpy.random as rd
from ElegantRL_learning.DQN.net import QNet,QNetDuel,QNetTwin,QNetTwinDuel
class AgentBase:
    '''
    属性
    learning rate
    state

    方法
    select_action
    explore_env:agent利用与环境互动，在此过程中产生用于训练的数据transition(s,a,r,s')
    update_net:agent从经验回放池中获取数据（minibatch条transition），利用这些数据更新网络
    save_load_model
    soft_update
    '''
    def __init__(self):
        self.states = None
        self.device = None
        self.action_dim = None
        self.if_on_policy = False
        self.explore_rate = 1.0  #探索率
        self.explore_noise = None  #探索噪声
        self.traj_list = None  #trajectory_list

        '''attribute'''
        self.explore_env = None
        self.get_obj_critic = None

    def init(self,net_dim,state_dim,action_dim,learning_rate=1e-4,
             if_per_or_gae=False,env_num=1,agent_id=0):
        self.action_dim = action_dim
        self.traj_list = [list() for _ in range(env_num)]
        self.device = torch.device(f"cuda:{agent_id}" if(torch.cuda.is_available() and (agent_id>=0)) else "cpu")

        if env_num > 1:
            self.explore_env = self.explore_vec_env
        else:
            self.explore_env = self.explore_one_env

    def select_actions(self,states):
        states = torch.as_tensor(states,dtype=torch.float32,device=self.device)
        actions = self.act(states)
        if rd.rand() < self.explore_rate:
            actions = (actions + torch.randn_like(actions)*self.explore_noise).clamp(-1,1)
        return actions.detach().cpu().numpy()

