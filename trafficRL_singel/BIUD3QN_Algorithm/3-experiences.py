#!/usr/bin/python
# -*- coding: UTF-8 -*-
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
from trafficRL_singel.BIUD3QN_Algorithm.biud3qn import main_biud3qn

from trafficRL_singel.DQN.dqn_agent import Agent as dqn_Agent
from trafficRL_singel.DQN.env import environment as dqn_environment
from trafficRL_singel.DQN.trafficSignal import TrafficSignal as dqn_TrafficSignal
from trafficRL_singel.DQN.replayBuffer import ReplayBuffer as dqn_ReplayBuffer
from trafficRL_singel.DQN.dqn import main_dqn

from trafficRL_singel.DQN.recordData import record_data
from tensorboardX import SummaryWriter
from trafficRL_singel.Fixed1.ft import ft_test2
from trafficRL_singel.DQN.unbalance_net.generateCarFlow import generate_routefile
from ElegantRL_learning.DQN.plot import plot
from trafficRL_singel.DQN.get_car_info import get_car_number_4

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)

if __name__ == '__main__':
    N = "29_非动态车流文件"

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
    capacity = 8000
    minibatch = 320

    # 智能体参数
    learning_rate = 0.001  # 学习率
    gamma = 0.9  # 折扣因子 0.98??
    epsilon = 0.1  # 探索率
    alpha = 0.02 # 0.025??

    # 神经网络参数
    biu_action_dim = 4
    biu_state_dim = 9
    biu_mid_dim = 3*biu_state_dim

    dqn_action_dim = 4
    dqn_state_dim = 5
    dqn_mid_dim = 3*dqn_state_dim

    Max_episodes = 5
    Max_steps = 7200
    C = 10

    # sumo 参数
    sumoBinary = 'sumo'  # 'sumo-gui'
    sumofile = '../DQN/unbalance_net/high_net.sumocfg'
    sumoConfig = [sumoBinary, '-c', sumofile, "--tripinfo-output", "tripinfo.xml"]

    print('----start time: ', datetime.datetime.now())

    # 计算奖励参数
    biu_c1 = 1
    biu_c2 = 1
    biu_c3 = 0.4

    # 初始化BIU D3QN交通灯，环境env，智能体agent
    biu_tlsID = 'gneJ0'
    biu_trafficlight = biu_TrafficSignal(biu_tlsID, biu_c1, biu_c2, biu_c3, 30)
    biu_env = biu_environment(inLanes, inEdges, outLanes, outEdges, biu_tlsID, biu_trafficlight)
    biu_buffer = biu_ReplayBuffer(capacity, minibatch)
    biu_agent = biu_Agent(biu_env, biu_state_dim, biu_mid_dim, biu_action_dim, replayBuffer=biu_buffer,
                          alpha=alpha, gamma=gamma, epsilon0=epsilon, learning_rate=learning_rate)

    dqn_c1 = 1
    dqn_c2 = 1
    dqn_c3 = 0.4
    # 初始化BIU D3QN交通灯，环境env，智能体agent
    dqn_tlsID = 'gneJ0'
    dqn_trafficlight = dqn_TrafficSignal(dqn_tlsID, dqn_c1, dqn_c2, dqn_c3, 30)
    dqn_env = dqn_environment(inLanes, inEdges, outLanes, outEdges, dqn_tlsID, dqn_trafficlight)
    dqn_buffer = dqn_ReplayBuffer(capacity, minibatch)
    dqn_agent = dqn_Agent(dqn_env, dqn_state_dim, dqn_mid_dim, dqn_action_dim, replayBuffer=dqn_buffer,
                          alpha=alpha, gamma=gamma, epsilon0=epsilon, learning_rate=learning_rate)

    # 车流文件相关信息 车辆总数 转向比等
    car_N = 3000  # 3000
    sumofile_name = '../DQN/unbalance_net/high_net.rou.xml'
    path2 = '../DQN/unbalance_net/high_net.rou.xml'
    L = 5  # 左转比例
    sRatio = 3  # 直行比例

    # 参数记录
    record = '''        
    车流文件数据：{0} 
    转弯比 {1}
        1、BIU_D3QN
        奖励设置：-c1*delay_time - c2*per_waiting_time
        奖励参数调整：c1={2} c2={3}
        参数设置:1) 探索率 {4} 动态改变
                2) 学习率 {5} 调整了学习率！！
                
                优化器：optimizer Adam
        改变状态 turing_rate 变为 turning_number
        要求全部跑完 限制相同动作不能重复3次以上
        mid_dim = 3 * input_dim  增大了经验缓冲池大小！
            
        2、DQN
        奖励设置：-c1*delay_time - c2*per_waiting_time
        奖励参数调整：c1={6} c2={7}
        参数设置:1) 探索率 {8} 固定值
                2) 学习率 {9}
                未增加k机制
        delay_time 修正为 1-(当前速度/max_speed) 测试
        左转：直行：右转 = 5：3: 2
        修正dqn的reward 
    '''.format(sumofile, L, biu_c1, biu_c2, epsilon, learning_rate, dqn_c1, dqn_c2, epsilon, learning_rate)

    with open('record.txt', 'a+') as f:
        f.writelines("文件名:" + N)
        f.writelines(record)

    # 待记录数据
    biu_per_reward_list = []
    biu_per_waiting_time_list = []
    biu_per_delay_time_list = []
    biu_per_queue_list = []
    biu_throughOutput_list = []
    biu_cz = []

    dqn_per_reward_list = []
    dqn_per_waiting_time_list = []
    dqn_per_delay_time_list = []
    dqn_per_queue_list = []
    dqn_throughOutput_list = []
    dqn_cz = []

    ft_waiting_time_list = []
    ft_delay_time_list = []
    ft_queue_list = []
    ft_throughOutput = []

    biu_net_update_times = 0
    dqn_net_update_times = 0

    # 开始 训练
    os.mkdir('policy_save/runs/{0}'.format(N))
    ft_waiting_time, rn, ft_queue, ft_delay_time = ft_test2(sumofile)
    for episode in range(Max_episodes):
        # 每一轮生成新车流文件
        # generate_routefile(sumofile_name, car_N, Max_steps, L, sRatio)
        # get_car_number_4(path2, episode)  # 获取本轮数据情况  # 修改生成一个直方图 每一轮啥啥数据的一个可视化

        # 运行fixedTime方案
        # ft_waiting_time, rn, ft_queue, ft_delay_time = ft_test2(sumofile)
        ft_waiting_time_list.append(round(ft_waiting_time / car_N, 2))
        ft_delay_time_list.append(round(ft_delay_time / car_N, 2))
        ft_queue_list.append(round(ft_queue / car_N, 2))
        ft_throughOutput.append(car_N - rn)
        print("fixed time {0} finished".format(episode))

        # biu 每个episode需要记录的变量
        per_reward, per_waiting_time, per_delay_time, per_queue, throughOutput, biu_net_update_times = \
            main_biud3qn(sumoConfig, biu_buffer, biu_agent, episode, Max_episodes, Max_steps, biu_net_update_times, C,
                         car_N,N)
        biu_per_reward_list.append(per_reward)
        biu_per_waiting_time_list.append(per_waiting_time)
        biu_per_delay_time_list.append(per_delay_time)
        biu_per_queue_list.append(per_queue)
        biu_throughOutput_list.append(throughOutput)
        biu_cz.append(ft_waiting_time_list[len(ft_waiting_time_list) - 1] -
                      biu_per_waiting_time_list[len(biu_per_waiting_time_list) - 1])

        #dqn 每个episode需要记录的变量

        per_reward, per_waiting_time, per_delay_time, per_queue, throughOutput, dqn_net_update_times = \
            main_dqn(sumoConfig, dqn_buffer, dqn_agent, episode, Max_episodes, Max_steps, dqn_net_update_times, C,
                     car_N)
        dqn_per_reward_list.append(per_reward)
        dqn_per_waiting_time_list.append(per_waiting_time)
        dqn_per_delay_time_list.append(per_delay_time)
        dqn_per_queue_list.append(per_queue)
        dqn_throughOutput_list.append(throughOutput)
        dqn_cz.append(ft_waiting_time_list[len(ft_waiting_time_list) - 1] -
                      dqn_per_waiting_time_list[len(dqn_per_waiting_time_list) - 1])


    writer = SummaryWriter('policy_save/runs/{0}/biu_d3qn_{0}'.format(N))
    for i in range(len(biu_per_reward_list)):
        writer.add_scalar('waiting_time', biu_per_waiting_time_list[i], global_step=i)
        writer.add_scalar('reward', biu_per_reward_list[i], global_step=i)
        writer.add_scalar('delay_time', biu_per_delay_time_list[i], global_step=i)
        writer.add_scalar('queue', biu_per_queue_list[i], global_step=i)
        writer.add_scalar('throughOutput', biu_throughOutput_list[i], global_step=i)
        writer.add_scalar('cz', biu_cz[i], global_step=i)

    writer = SummaryWriter('policy_save/runs/{0}/dqn_{0}'.format(N))
    for i in range(len(dqn_per_reward_list)):
        writer.add_scalar('waiting_time', dqn_per_waiting_time_list[i], global_step=i)
        writer.add_scalar('reward', dqn_per_reward_list[i], global_step=i)
        writer.add_scalar('delay_time', dqn_per_delay_time_list[i], global_step=i)
        writer.add_scalar('queue', dqn_per_queue_list[i], global_step=i)
        writer.add_scalar('throughOutput', dqn_throughOutput_list[i], global_step=i)
        writer.add_scalar('cz', dqn_cz[i], global_step=i)
    from tensorboardX import SummaryWriter
    writer = SummaryWriter('policy_save/runs/{0}/ft_example_{0}'.format(N))
    for i in range(len(ft_waiting_time_list)):
        writer.add_scalar('waiting_time', ft_waiting_time_list[i], global_step=i)
        writer.add_scalar('delay_time', ft_delay_time_list[i], global_step=i)
        writer.add_scalar('queue', ft_queue_list[i], global_step=i)
        writer.add_scalar('throughOutput', ft_throughOutput[i], global_step=i)

    # 存数据
    fname = 'policy_save/runs/{0}/biu_data_{0}.xlsx'.format(N)
    biu_data_list = {'reward': biu_per_reward_list,
                     'waiting_time': biu_per_waiting_time_list,
                     'delay_time': biu_per_delay_time_list,
                     'queue': biu_per_queue_list,
                     'throughOutput': biu_throughOutput_list,
                     'cz': biu_cz}
    record_data(fname, biu_data_list)
    fname2 = 'policy_save/runs/{0}/ft_data_{0}.xlsx'.format(N)
    ft_data_list = {'waiting_time': ft_waiting_time_list,
                    'delay_time': ft_delay_time_list,
                    'queue': ft_queue_list,
                    'throughOutput': ft_throughOutput}
    record_data(fname2, ft_data_list)

    fname3 = 'policy_save/runs/{0}/dqn_data_{0}.xlsx'.format(N)
    dqn_data_list = {'reward': dqn_per_reward_list,
                     'waiting_time': dqn_per_waiting_time_list,
                     'delay_time': dqn_per_delay_time_list,
                     'queue': dqn_per_queue_list,
                     'throughOutput': dqn_throughOutput_list,
                     'cz': dqn_cz
                     }
    record_data(fname3, dqn_data_list)

    # 绘图
    title_list = {'reward': [biu_per_reward_list, dqn_per_reward_list],
                  'waiting_time': [ft_waiting_time_list, biu_per_waiting_time_list, dqn_per_waiting_time_list],
                  'delay_time': [ft_delay_time_list, biu_per_delay_time_list, dqn_per_delay_time_list],
                  'queue': [ft_queue_list, biu_per_queue_list, dqn_per_queue_list],
                  'throughOutput': [ft_throughOutput, biu_throughOutput_list, dqn_throughOutput_list]
                  }

    plt_path = "../../trafficRL_singel/BIUD3QN_Algorithm/policy_save/runs/{0}".format(N)
    for tag, data in title_list.items():
        if tag != 'reward':
            plot(tag, plt_path, 'fixedtime', 'biu_d3qn', 'dqn', fixedtime=data[0], biu_d3qn=data[1], dqn=data[2])
        else:
            plot(tag, plt_path, 'biu_d3qn', 'dqn', biu_d3qn=data[0], dqn=data[1])
