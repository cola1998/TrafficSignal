import os
import sys
import torch
import traci
import datetime
from collections import namedtuple
from trafficRL_singel.BIUD3QN_Algorithm.d3qn_agent import Agent as biu_Agent
from trafficRL_singel.BIUD3QN_Algorithm.env import environment as biu_environment
from trafficRL_singel.BIUD3QN_Algorithm.trafficSignal import TrafficSignal as biu_TrafficSignal
from trafficRL_singel.BIUD3QN_Algorithm.replayBuffer import ReplayBuffer as  biu_ReplayBuffer

from trafficRL_singel.DQN.dqn_agent import Agent as dqn_Agent
from trafficRL_singel.DQN.env import environment as dqn_environment
from trafficRL_singel.DQN.trafficSignal import TrafficSignal as dqn_TrafficSignal
from trafficRL_singel.DQN.replayBuffer import ReplayBuffer as dqn_ReplayBuffer

from trafficRL_singel.DQN.recordData import record_data,record_data_ft
from tensorboardX import SummaryWriter
from trafficRL_singel.Fixed1.ft import ft_test2
from trafficRL_singel.DQN.unbalance_net.generateCarFlow import generate_routefile
from trafficRL_singel.DQN.get_car_info import get_car_number_4

'''
dqn : dqn_buffer dqn_agent dqn_trafficSignal 
d3qn : d3qn_buffer d3qn_agent d3qn_trafficSignal
需要存储的数据
reward delay_time waiting_time queue throughOutput total_loss loss_list
'''
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)

if __name__ == '__main__':
    N = ""

    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    # 环境参数
    inLanes = [
        'gneE0_0', 'gneE0_1', 'gneE0_2', 'gneE0_3',
        'gneE1_0', 'gneE1_1', 'gneE1_2', 'gneE1_3',
        'gneE2_0', 'gneE2_1', 'gneE2_2', 'gneE2_3',
        'gneE3_0', 'gneE3_1', 'gneE3_2', 'gneE3_3',
    ]
    outLanes = [
        '-gneE0_0', '-gneE0_1', '-gneE0_2', '-gneE0_3',
        '-gneE1_0', '-gneE1_1', '-gneE1_2', '-gneE1_3',
        '-gneE2_0', '-gneE2_1', '-gneE2_2', '-gneE2_3',
        '-gneE3_0', '-gneE3_1', '-gneE3_2', '-gneE3_3',
    ]
    inEdges = ['gneE0', 'gneE1', 'gneE2', 'gneE3']
    outEdges = ['-gneE0', '-gneE1', '-gneE2', '-gneE3']

    # 缓冲池参数
    capacity = 1000
    minibatch = 32

    # 智能体参数
    learning_rate = 0.005  # 学习率
    gamma = 0.9  # 折扣因子
    epsilon = 0.5  # 探索率
    alpha = 0.3

    # 神经网络参数
    action_dim = 4
    state_dim = 9
    mid_dim = 3

    Max_episodes = 100
    Max_steps = 7200
    C = 10

    # sumo 参数
    sumoBinary = 'sumo'  # 'sumo-gui'
    sumofile = '../DQN/unbalance_net/net.sumocfg'
    sumoConfig = [sumoBinary, '-c', sumofile, "--tripinfo-output", "tripinfo.xml"]

    print('----start time: ', datetime.datetime.now())

    # 计算奖励参数
    biu_c1 = 0
    biu_c2 = 0.5
    biu_c3 = 0.4

    # 初始化BIU D3QN交通灯，环境env，智能体agent
    biu_tlsID = 'gneJ0'
    biu_trafficlight = biu_TrafficSignal(biu_tlsID, biu_c1, biu_c2, biu_c3, 30)
    biu_env = biu_environment(inLanes, inEdges, outLanes, outEdges, biu_tlsID, biu_trafficlight)
    biu_buffer = biu_ReplayBuffer(capacity, minibatch)
    biu_agent = biu_Agent(biu_env, state_dim, mid_dim, action_dim, replayBuffer=biu_buffer,
                  alpha=alpha, gamma=gamma, epsilon0=epsilon, learning_rate=learning_rate)

    dqn_c1 =
    dqn_c2 =
    dqn_c3 =
    # 初始化BIU D3QN交通灯，环境env，智能体agent
    dqn_tlsID = 'gneJ0'
    dqn_trafficlight = dqn_TrafficSignal(dqn_tlsID, dqn_c1, dqn_c2, dqn_c3, 30)
    dqn_env = dqn_environment(inLanes, inEdges, outLanes, outEdges, dqn_tlsID, dqn_trafficlight)
    dqn_buffer = dqn_ReplayBuffer(capacity, minibatch)
    dqn_agent = dqn_Agent(dqn_env, state_dim, mid_dim, action_dim, replayBuffer=dqn_buffer,
                  alpha=alpha, gamma=gamma, epsilon0=epsilon, learning_rate=learning_rate)

    # 车流文件相关信息 车辆总数 转向比等
    car_N = 3000  # 3000
    sumofile_name = '../DQN/unbalance_net/net_3000.rou.xml'
    path2 = '../DQN/unbalance_net/net_3000.rou.xml'
    L = 6 # 左转比例
    sRatio = 3 # 直行比例

    # 参数记录
    record = '''        
    数据：{0}
    奖励设置：-c1*delay_time - c2*per_waiting_time
    奖励参数调整：c1={1} c2={2}
    参数设置:1) 探索率 {3} 动态改变
            2) 学习率 {4}
            未增加k机制
            优化器：optimizer Adam
            查看perloss的变化
            转弯比{5}
    '''.format(sumofile, c1, c2, epsilon, learning_rate, L)
    with open('record.txt', 'a+') as f:
        f.writelines("文件名:" + N)
        f.writelines(record)

    # 待记录数据
    biu_per_reward_list = []
    biu_per_waiting_time_list = []
    biu_per_delay_time_list = []
    biu_per_queue_list = []
    biu_throughOutput_list = []

    dqn_per_reward_list = []
    dqn_per_waiting_time_list = []
    dqn_per_delay_time_list = []
    dqn_per_queue_list = []
    dqn_throughOutput_list = []

    ft_waiting_time_list = []
    ft_delay_time_list = []
    ft_queue_list = []
    ft_throughOutput = []

    biu_net_update_times = 0
    dqn_net_update_times = 0

    # 开始 训练
    for episode in range(Max_episodes):
        main_biud3qn()
        main_dqn()