import torch
import gym
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
from numpy import *
import matplotlib.pyplot as plt

'''
基于gym库学习强化学习代码的写法  加油 完成日期 2021.11.11日
https://www.w3cschool.cn/pytorch/pytorch-1zvj3bpn.html
'''
env = gym.make('CartPole-v0').unwrapped
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))  # 命名元组


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0  # 用于记录当前存储的位置

    def push(self, *args):
        '''
        存储transition功能
        :param args:
        :return:
        '''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        '''
        采样
        :param batch_size: 采样大小
        :return:  返回一个batch_size大小的列表 存放self.memeory产生的batch_size条随机且唯一的transition数据
        '''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):  # nn.Module is standard PyTorch Network
    def __init__(self, state_dim, mid_dim, outputs, h, w):
        '''
        相当于create model
        :param mid_dim: 中间层神经元数
        :param state_dim: 状态层神经元数 输入
        :param action_dim: 动作层神经元数 输出
        '''
        super().__init__()  # 第一句话，调用父类的构造函数
        # self.net = nn.Sequential(
        #     nn.Linear(state_dim, mid_dim),
        #     nn.ELU(),
        #     nn.Linear(mid_dim, mid_dim),
        #     nn.ELU(),
        #     nn.Linear(mid_dim, mid_dim),
        #     nn.ELU(),
        #     nn.Linear(mid_dim, action_dim)
        # )

        self.conv1 = nn.Conv2d(state_dim, mid_dim, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(mid_dim)
        self.conv2 = nn.Conv2d(mid_dim, mid_dim * 2, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(mid_dim * 2)
        self.conv3 = nn.Conv2d(mid_dim * 2, mid_dim * 2, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(mid_dim * 2)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0),-1))  # 计算Q-value  直接返回action-dim的tensor


state_dim = 3
mid_dim = 16

# 获取输入

# 训练
batch_size = 128
gamma = 0.999
eps_start = 0.9
eps_end = 0.05
eps_decay = 200
target_update = 10

init_screen = get_screen()  # 获取状态
_,_,screen_height,screen_width = init_screen.shape

n_action = env.action_space.n # n_action action_dim
policy_net = Net()
target_net = Net()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

step_done = 0
def select_action(state):
    global step_done
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1*step_done/eps_decay)

    step_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            policy_net(state).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(n_action)]], device=device, dtype=torch.long)


episode_durations = []
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# 训练模型
def optimize_model():
    '''
    首先进行随机采样,连接成一个张量
    计算Q V
    :return:
    '''
    if len(memory) <batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool
                                  )
    # non_final_mask = tensor([True, True])
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 计算Q(s_t,a) 然后选择采取行动的列
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 计算下一个状态的v
    # 基于旧网络计算  选择max(1)[0]的最佳奖励。
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # 计算期望q值
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # 计算损失
    loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 优化模型
    optimizer.zero_grad()
    loss.backward()

    for param in policy_net.parameters():
        param.grad.data.clamp_(-1,1)  # 将input中的元素限制在[min,max]范围内并返回一个Tensor
    optimizer.step()


#  主训练过程
max_episodes = 50
max_steps = 100
for i_episode in range(max_episodes):
    state = 0
    for t in max_steps:
        action = select_action(state) # 选择动作

        _,reward,done,_ = env.step(action.item()) # 执行动作
        reward = torch.tensor([reward],device=device)

        next_state = get_state() #last_screen - current_screen
        memory.push(state,action,next_state,reward)
        state = next_state

        optimize_model()
        if done:
            episode_durations.append(t+1)
            break
    if i_episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
