import os
import sys
import torch
import traci
import datetime
from collections import namedtuple
from trafficRL_singel.BIU_DQN_Algorithm.dqn_agent2 import Agent
from trafficRL_singel.BIU_DQN_Algorithm.env import environment
from trafficRL_singel.BIU_DQN_Algorithm.trafficSignal import TrafficSignal
from trafficRL_singel.BIU_DQN_Algorithm.replayBuffer import ReplayBuffer
from trafficRL_singel.net.netFile.generateCarFlow import generate_routefile
from ElegantRL_learning.DQN.plot import plot_BIUDQN
from tensorboardX import SummaryWriter

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)

if __name__ == "__main__":
    # 初始化
    N = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward','mask'))  # 命名元组

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
    capacity = 2000
    minibatch = 100

    # 初始化各个参数
    learning_rate = 0.9  # 学习率
    gamma = 0.9  # 折扣因子
    epsilon = 0.5  # 探索率
    alpha = 0.3

    action_dim = 4
    state_dim = 9
    mid_dim = 3

    Max_episodes = 20
    Max_steps = 5400
    C = 10
    reward_list = []
    waiting_time_list = []
    loss_list = []
    remain_car_numbers = []
    sumoBinary =  'sumo' #'sumo-gui'
    sumoConfig = [sumoBinary, '-c', '../net/netFile/dynamicNet.sumocfg', "--tripinfo-output", "tripinfo.xml"]

    print('----start time: ', datetime.datetime.now())

    # 交通灯参数
    c1 = 0.4
    c2 = 0.4
    c3 = 0.4
    tlsID = 'gneJ0'
    # 初始化交通灯，环境env，智能体agent
    trafficlight = TrafficSignal(tlsID, c1, c2, c3, 30)
    env = environment(inLanes, inEdges, outLanes, outEdges, tlsID, trafficlight)
    buffer = ReplayBuffer(capacity, minibatch)
    agent = Agent(env, state_dim, mid_dim, action_dim, replayBuffer=buffer,
                  alpha=alpha, gamma=gamma, epsilon0=epsilon, learning_rate=learning_rate)
    dummy_input = torch.rand(1, state_dim)
    with SummaryWriter('runs/graph') as w:
        w.add_graph(agent.cri, dummy_input)
    # 一些关于车流的参数
    l_N = 3000  # 低车流量
    l_file_name = '../net/netFile/dynamic_low_net.rou.xml'
    # m_N = 8000  # 中车流量
    # m_file_name = 'dynamic_mid_net.rou.xml'
    # h_N = 10000  # 高车流量
    # h_file_name = 'dynamic_high_net.rou.xml'

    # 不同的转弯比例
    lowL = 2
    midL = 3
    highL = 4
    sRatio = 5
    net_update_times = 0

    for episode in range(Max_episodes):
        # 考虑是否每一轮都需要重新生成车流文件
        # generate_routefile(l_file_name, l_N, Max_steps, highL, sRatio)  #每轮动态产生

        R = 0
        waiting_time = 0

        # 处理traci接口
        traci.start(sumoConfig)
        tlsID = traci.trafficlight.getIDList()[0]
        traci.trafficlight.setPhase(tlsID, 0)  # 信号灯相位从0开始
        traci.simulationStep()
        current_state = agent.get_state()

        step = 0
        while traci.simulation.getMinExpectedNumber() > 0 and step < Max_steps:
            res = agent.identifyFlow()
            if res != None:
                flowTag = res[0]   # 'k' and 'ra'
                flowIndex = int(res[1])  # 0,1,2,3
            else:
                flowTag = 'ba'
                flowIndex = 0
            current_state = torch.tensor(current_state, dtype=torch.float32)
            current_state = current_state.view(1, state_dim)
            action = agent.select_action(current_state)
            step, reward,waiting_time_t = agent.take_action(action, step, flowTag,flowIndex)
            next_state = agent.get_state()
            if step >= Max_steps:
                buffer.append_buffer(current_state, action, next_state, reward, 0.0)
            else:
                buffer.append_buffer(current_state, action, next_state, reward, gamma)
            current_state = next_state

            loss = agent.optimize_model()
            if loss != None:
                loss_list.append(loss)
            R += reward
            waiting_time += waiting_time_t
            net_update_times += 1
            if net_update_times % C == 0:
                agent.optimize_target_model()

        print("episode= {0} 结束时traci.simulation.getMinExpectedNumber={1} 和 step={2}".format(episode,
                                                                                            traci.simulation.getMinExpectedNumber(),
                                                                                            step))
        remain_car_numbers.append(traci.simulation.getMinExpectedNumber())
        reward_list.append(R)
        waiting_time_list.append(waiting_time)
        traci.close()

    print("reward_list : ", reward_list)
    print("waiting_time_list : ", waiting_time_list)

    writer = SummaryWriter('policy_save/runs/dqn_{0}'.format(N))
    for i in range(len(waiting_time_list)):
        writer.add_scalar('waiting_time', waiting_time_list[i], global_step=i)
        writer.add_scalar('reward', reward_list[i], global_step=i)
        writer.add_scalar('remain_car_number',remain_car_numbers[i],global_step=i)
    writer = SummaryWriter('policy_save/runs/loss_dqn_{0}'.format(N))
    for i in range(len(loss_list)):
        writer.add_scalar('loss', loss_list[i], global_step=i)
    plot_BIUDQN(Max_episodes, reward_list, waiting_time_list)