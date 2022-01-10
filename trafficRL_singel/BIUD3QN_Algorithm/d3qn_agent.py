import numpy as np
import random
import torch
import torch.optim as optim
from ElegantRL_learning.DQN.net import QNet, QNetDuel, QNetTwin, QNetTwinDuel
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward','mask'))  # 命名元组


class Agent():
    def __init__(self, env, state_dim, mid_dim, action_dim, replayBuffer, alpha=None, gamma=None, epsilon0=None,
                 learning_rate=None, k=None, ra=None):
        self.learning_rate = learning_rate if learning_rate is not None else 0.9
        self.gamma = gamma if gamma is not None else 0.9
        self.epsilon = 1  # 有的需要 有的不需要
        self.alpha = alpha if alpha is not None else 0.8

        self.state_dim = state_dim
        self.mid_dim = mid_dim
        self.action_dim = action_dim

        self.cri = QNetTwinDuel(self.state_dim, self.mid_dim, self.action_dim)
        self.cri_target = QNetTwinDuel(self.state_dim, self.mid_dim, self.action_dim)
        self.act = self.act_target = None

        self.env = env  # 每个智能体有一个自己的环境对应  就是environment对象 然后从环境中获取环境的相关信息 状态
        self.replayBuffer = replayBuffer  # 对应经验缓冲池
        # optim.RMSprop
        self.optimizer = optim.Adam(self.cri.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.SmoothL1Loss()
        self.soft_update_tau = 2**-8
        self.k = 0.55 if k is None else k
        self.ra = 0.40 if ra is None else ra

    def get_state(self,*args):
        return self.env.get_state(*args)

    def select_action(self, state,epsilon0 = 0.0):
        self.epsilon = max(self.epsilon * 0.99,0.05)

        if np.random.rand() <= self.epsilon:
            return torch.as_tensor([random.randrange(self.action_dim)])
        else:
            with torch.no_grad():
                action = self.cri(state)
            return action.argmax(dim=1)

    def take_action(self, current_action, step, flowTag, flowIndex):
        '''
            传入action，n_step   要执行的动作 和要执行多少步
            然后判断是否切换相位
            如果切换相位先执行last_action+1相位的3step
            判断action
            如果是0，1，2，3 自动*2 然后执行n_step步
            如果是4，5 执行n_step=12步
        '''
        delay_time, queue = 0,0
        waiting_time_t = 0
        if flowTag != 'ba':
            if flowTag == 'k':
                if flowIndex == current_action:
                    n_step = 15
                    now_step = self.env.step(flowIndex, step, n_step)
                    waiting_time_t += self.env.get_total_waiting_time()
                    delay_time += self.env.get_delay_time(now_step)
                    queue += self.env.get_total_lanes_queue()
                else:
                    # n_step = 12
                    # now_step = self.env.step(flowIndex, step, n_step)
                    now_step = step
                n_step = 30
                now_step = self.env.step(current_action, now_step, n_step)  # 判断没有切换相位 执行30步
                waiting_time_t += self.env.get_total_waiting_time()
                delay_time += self.env.get_delay_time(now_step)
                queue += self.env.get_total_lanes_queue()
            else:
                n_step = 30
                if flowIndex == 0 or flowIndex == 2:
                    action = 4
                    now_step = self.env.step(action, step, n_step)
                    waiting_time_t += self.env.get_total_waiting_time()
                    delay_time += self.env.get_delay_time(now_step)
                    queue += self.env.get_total_lanes_queue()
                else:
                    action = 5
                    now_step = self.env.step(action, step, n_step)
                    waiting_time_t += self.env.get_total_waiting_time()
                    delay_time += self.env.get_delay_time(now_step)
                    queue += self.env.get_total_lanes_queue()
                n_step = 30
                now_step = self.env.step(current_action, now_step, n_step)  # 判断切换相位了，自动执行3步 + n_step步
                waiting_time_t += self.env.get_total_waiting_time()
                delay_time += self.env.get_delay_time(now_step)
                queue += self.env.get_total_lanes_queue()
            reward = self.env.get_reward(now_step)
        else:
            now_step = self.env.step(current_action, step,30)
            reward = self.env.get_reward(now_step)
            waiting_time_t += self.env.get_total_waiting_time()
            delay_time += self.env.get_delay_time(now_step)
            queue += self.env.get_total_lanes_queue()
        return now_step, reward, waiting_time_t,delay_time,queue

    def identifyFlow(self):
        k_list = []
        q_list = []
        for edge in self.env.inEdges:
            q_list.append(self.env.get_edge_queue(edge))

        if sum(q_list) == 0:
            k = 0
        else:
            for q in q_list:
                k_list.append(round(q / sum(q_list), 2))
            k = max(k_list)

        ra_list = []
        ls, ss, rs = self.env.get_turn_number_inEdges()

        for i in range(len(ls)):
            if (ls[i] + ss[i] + rs[i]) != 0:
                Rl = round(ls[i] / (ls[i] + ss[i] + rs[i]), 2)
                ra_list.append(Rl)
            else:
                ra_list.append(0)
        ra = max(ra_list)
        # if k >= self.k:
        #     return ["k", k_list.index(k)]

        if ra >= self.ra:
            return ["ra", ra_list.index(ra)]
        else:
            return None


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
        try:
            batch = Transition(*zip(*transitions))
        except TypeError as e:
            print("出现typeError 详情：{0}".format(e))
            print("打印batch：",Transition(*zip(*transitions)))
            return

        '''
        更新网络这一步 如何做
        self.criterion = torch.nn.SmoothL1Loss()
        self.cri() self.cri_target() self.cri.get_q1_q2() self.cri_target.get_q1_q2()
        mask = 0.0 if done else gamma
        q1,q2 = self.cri.get_q1_q2(state)  # (tensor([[-0.2617, -0.2617, -0.2617, -0.2617]]), tensor([[0.3207, 0.3207, 0.3207, 0.3207]]))
        next_q = torch.min(cri_target.get_q1_q2(next_s)).max()
    
        q_label = reward + mask * next_q 
        q1 ,q2 = self.act.get_q1_q2(state) ????
        obj_critic = self.criterion(q1,q_label) + self.criterion(q2,q_label)
        
        optimizer 更新
        self.optimizer.zero_grad()
        obj_critic.requires_grad_(True)
        obj_critic.backward()
        self.optimizer.step()
        
        soft_update  更新目标网络参数  使用软更新 θ = τ*θ' + （1-τ）*θ
        for tar, cur in zip(self.cri_target.parameters(),self.cri.parameters()):
            tar = cur * tau + tar * (1-tau)
        '''
        q_label = []
        q_eval = []  # q现实值
        q_g = []
        for i in range(len(batch.state)):
            state = torch.as_tensor(batch.state[i],dtype=torch.float)
            next_state = torch.as_tensor(batch.next_state[i],dtype=torch.float)
            mask = batch.mask[i]
            reward = batch.reward[i]

            state_value = self.cri(state).max(0)[0]
            # q_g.append(state_value)
            q = self.cri_target(state).max(0)[0]
            # q_eval.append(q)
            t = torch.cat([t.unsqueeze(0) for t in self.cri.get_q1_q2(next_state)],0)
            max_eval4next = t.max(0)[0].unsqueeze(0).max(1)[1]  # 对应最大值的index
            next_state_value = self.cri_target(next_state)[max_eval4next]

            ql = state_value + self.alpha*(reward + mask * next_state_value)
            q_label.append(ql)
        q_label = torch.cat([ql.unsqueeze(0) for ql in q_label],0)
        # 计算损失
        q1_list,q2_list = [],[]

        for state in batch.state:
            q1,q2 = self.cri.get_q1_q2(state)

            q1_list.append(q1)
            q2_list.append(q2)
        q1_list = torch.cat([q1 for q1 in q1_list],0)
        q2_list = torch.cat([q2 for q2 in q2_list],0)

        loss = self.criterion(q1_list, q_label) + self.criterion(q2_list,q_label)
        # td_error 目标网络产生的Q值[Q现实值] - 在线网络产生的Q值[Q估计值]
        # 优化模型
        self.optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        self.optimizer.step()
        # print("loss:",loss.item())
        return loss.item()

    def optimize_target_model(self):
        for tar,cur in zip(self.cri_target.parameters(),self.cri.parameters()):
            tar.data.copy_(cur.data*self.soft_update_tau + cur.data*(1-self.soft_update_tau))
