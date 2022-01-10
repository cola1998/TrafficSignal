'''
存储可能会用到的神经网络
有三种Q网络 Policy网络(Actor)  Value网络(Critic)
Q网络
class QNet()
class QNetDuel() dueling dqn
class QNetTwin() double dqn
class QNetTwinDuel()  d3qn

Policy网络
class Actor
class ActorPPO
class ActorSAC

Value网络
class Critic
class CriticAdv
class CriticTwin
'''
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict


class QNet(nn.Module):  # nn.Module is standard PyTorch Network
    def __init__(self,state_dim, mid_dim,  action_dim):
        '''
        相当于create model
        :param mid_dim: 中间层神经元数
        :param state_dim: 状态层神经元数 输入
        :param action_dim: 动作层神经元数 输出
        LeakyReLU(negative_slope=0.01, inplace=False)
        '''
        super().__init__()  # 第一句话，调用父类的构造函数
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ELU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ELU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ELU(),
            nn.Linear(mid_dim, action_dim)
        )

    def forward(self, state):
        res = self.net(state)
        return res  # 计算Q-value  直接返回一个8维的action列表[]



class QNetDuel(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_state = nn.Sequential(
            OrderedDict(
                [
                    ("dense1", nn.Linear(state_dim, mid_dim)),
                    ("relu1", nn.ReLU()),
                    ("dense2", nn.Linear(mid_dim, mid_dim)),
                    ("relu2", nn.ReLU())
                ]
            )
        )
        self.net_adv = nn.Sequential(
            OrderedDict([
                ("dense3", nn.Linear(mid_dim, mid_dim)),
                ("hardswish1", nn.Hardswish()),
                ("dense4", nn.Linear(mid_dim, 1))
            ])
        )
        self.net_val = nn.Sequential(
            OrderedDict([
                ("dense5", nn.Linear(mid_dim, mid_dim)),
                ("hardswish2", nn.Hardswish()),
                ("dense6", nn.Linear(mid_dim, action_dim))
            ])
        )

    def forward(self, state):
        t_tmp = self.net_state(state)
        q_adv = self.net_adv(t_tmp)
        q_val = self.net_val(t_tmp)
        return q_adv + q_val - q_val.mean(dim=1, keepdim=True)


class QNetTwin(nn.Module):  # double dqn
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_state = nn.Sequential(
            OrderedDict([
                ("dense1", nn.Linear(state_dim, mid_dim)),
                ("relu1", nn.ReLU()),
                ("dense2", nn.Linear(mid_dim, mid_dim)),
                ("relu2", nn.ReLU())
            ])
        )
        self.net_q1 = nn.Sequential(
            OrderedDict([
                ("dense3", nn.Linear(mid_dim, mid_dim)),
                ("hardswish1", nn.Hardswish()),
                ("dense4", nn.Linear(mid_dim, action_dim))
            ])
        )
        self.net_q2 = nn.Sequential(
            OrderedDict([
                ("dense5", nn.Linear(mid_dim, mid_dim)),
                ("hardswish2", nn.Hardswish()),
                ("dense6", nn.Linear(mid_dim, action_dim))
            ])
        )

    def forward(self, state):
        t_tmp = self.net_state(state)
        return self.net_q1(t_tmp)

    def get_q1_q2(self, state):
        tmp = self.net_state(state)
        return self.net_q1(tmp), self.net_q2(tmp)


class QNetTwinDuel(nn.Module):
    def __init__(self,state_dim,mid_dim,action_dim):
        super().__init__()
        self.net_state = nn.Sequential(
            OrderedDict([
                ("dense1", nn.Linear(state_dim, mid_dim)),
                ("relu1", nn.ReLU()),
                ("dense2", nn.Linear(mid_dim, mid_dim)),
                ("relu2", nn.ReLU())
            ])
        )
        self.net_adv1 = nn.Sequential(
            OrderedDict([
            ("dense3", nn.Linear(mid_dim, mid_dim)),
            ("hardswish1", nn.Hardswish()),
            ("dense4", nn.Linear(mid_dim, 1))
        ]))
        self.net_adv2 = nn.Sequential(
            OrderedDict([
            ("dense5", nn.Linear(mid_dim, mid_dim)),
            ("hardswish2", nn.Hardswish()),
            ("dense6", nn.Linear(mid_dim, 1))
        ]))
        self.net_val1 = nn.Sequential(
            OrderedDict([
                ("dense7", nn.Linear(mid_dim, mid_dim)),
                ("hardswish3", nn.Hardswish()),
                ("dense8", nn.Linear(mid_dim, action_dim))
        ]))
        self.net_val2 = nn.Sequential(
            OrderedDict([
                ("dense9", nn.Linear(mid_dim, mid_dim)),
                ("hardswish4", nn.Hardswish()),
                ("dense10", nn.Linear(mid_dim, action_dim))
        ]))

    def forward(self,state):
        t_tmp = self.net_state(state)
        q_adv = self.net_adv1(t_tmp)
        q_val = self.net_val1(t_tmp)
        # print(q_val)
        return q_adv+q_val-q_val.mean(dim=0)

    def get_q1_q2(self,state):
        tmp = self.net_state(state)
        adv1 = self.net_adv1(tmp)
        val1 = self.net_val1(tmp)
        q1 = adv1 + val1-val1.mean(dim=0)

        adv2 = self.net_adv2(tmp)
        val2 = self.net_val2(tmp)
        q2 = adv2 + val2 - val2.mean(dim=0)

        return [q1,q2]

# if __name__ == '__main__':
#     '''
#     pytorch 必须在init初始化网络结构  forward中做feed forward网络的前馈
#     听说 pytorch的训练需要自己写？
#     '''
#     # 初始化 模型的类
#     net = QNet(13, 7, 1)
#     # 选择 损失函数和优化器
#     criterion = torch.nn.MSELoss(reduction='sum')
#     optimizer = torch.optim.SGD(net.parameters(),lr=1e-4)
#     y = torch.tensor(5.2)
#     # 可以开始训练了
#     state = torch.tensor([20,15,14,12,20.00,50,7,4.98,0.4,0.8,0.1,0,5])
#     for t in range(500):
#         y_pred = net(state)
#
#         loss = criterion(y_pred,y)   # 计算损失函数 但是强化学习没有y呀？？？
#         print(t,y_pred,loss)
#         optimizer.zero_grad() # 梯度置零
#         loss.backward()
#         optimizer.step()

    # print("action:",action)  # 返回预测的action看看
    # print(int(action.argmax()))   # 找出预测action中最大值对应的index
