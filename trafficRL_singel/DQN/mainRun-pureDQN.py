import os
import sys
import torch
import traci
import datetime
from collections import namedtuple
from trafficRL_singel.DQN.dqn_agent import Agent
from trafficRL_singel.DQN.env import environment
from trafficRL_singel.DQN.trafficSignal import TrafficSignal
from trafficRL_singel.DQN.replayBuffer import ReplayBuffer
from trafficRL_singel.DQN.recordData import record_data
from trafficRL_singel.net.netFile.generateCarFlow import generate_routefile
from ElegantRL_learning.DQN.plot import plot_BIUDQN
from tensorboardX import SummaryWriter

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)

if __name__ == "__main__":
    # 初始化
    N = '13_high'

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
    capacity = 1000
    minibatch = 32

    # 初始化各个参数
    learning_rate = 0.1  # 学习率
    gamma = 0.9  # 折扣因子
    epsilon = 0.5  # 探索率
    alpha = 0.3

    action_dim = 4
    # state_dim = 9
    state_dim = 5
    mid_dim = 3

    Max_episodes = 25
    Max_steps = 7200
    C = 10

    sumoBinary = 'sumo' #'sumo-gui'
    sumofile = './balance_net/high_net.sumocfg'
    # sumoConfig = [sumoBinary, '-c', './balance_net/high_net.sumocfg', "--tripinfo-output", "tripinfo.xml"]
    sumoConfig = [sumoBinary, '-c', sumofile , "--tripinfo-output", "tripinfo.xml"]
    print('----start time: ', datetime.datetime.now())

    # 交通灯参数
    c1 = 0.4
    c2 = 1
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
    car_N = 3000
    record = '''        
                数据：{0}
                奖励设置：-c1*delay_time - c2*per_waiting_time
                奖励参数调整：c1={1} c2={2}
                参数设置:1) 探索率 {3} 固定值
                2) 学习率 {4}
                未增加k机制
                '''.format(sumofile, c1, c2, epsilon, learning_rate)
    with open('record.txt', 'a+') as f:
        f.writelines("文件名:" + N)
        f.writelines(record)
    per_reward_list = []
    per_waiting_time_list = []
    per_delay_time_list = []
    per_queue_list = []
    per_queue_list2 = []
    loss_list = []
    remain_car_numbers = []

    net_update_times = 0
    for episode in range(Max_episodes):
        print("episode:",episode)
        # 考虑是否每一轮都需要重新生成车流文件
        R = 0
        waiting_time = 0
        queue = 0
        delay_time = 0

        # 处理traci接口
        traci.start(sumoConfig)
        tlsID = traci.trafficlight.getIDList()[0]
        traci.trafficlight.setPhase(tlsID, 0)  # 信号灯相位从0开始
        traci.simulationStep()
        current_state = agent.get_state()

        step = 0
        while traci.simulation.getMinExpectedNumber() > 0 and step < Max_steps:
            current_state = torch.tensor(current_state, dtype=torch.float32)
            current_state = current_state.view(1, state_dim)
            action = agent.select_action(current_state)
            step, reward,waiting_time1, delay_time1,queue1 = agent.take_action(action, step)
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
            waiting_time += waiting_time1
            queue += queue1
            delay_time += delay_time1
            net_update_times += 1
            if net_update_times % C == 0:
                agent.optimize_target_model()


        per_reward_list.append(round(R/net_update_times,2))
        per_waiting_time_list.append(round(waiting_time/car_N,2))
        per_delay_time_list.append(round(delay_time/car_N,2))
        per_queue_list.append(round(queue/car_N,2))
        remain_car_numbers.append(traci.simulation.getMinExpectedNumber())
        traci.close()

    os.mkdir('policy_save/new_runs/{0}'.format(N))
    writer = SummaryWriter('policy_save/new_runs/{0}/pure_dqn_{0}'.format(N))
    for i in range(len(per_reward_list)):
        writer.add_scalar('waiting_time', per_waiting_time_list[i], global_step=i)
        writer.add_scalar('reward', per_reward_list[i], global_step=i)
        writer.add_scalar('delay_time', per_delay_time_list[i], global_step=i)
        writer.add_scalar('queue', per_queue_list[i], global_step=i)
        writer.add_scalar('remain_car_number',remain_car_numbers[i],global_step=i)
    writer = SummaryWriter('policy_save/new_runs/{0}/loss_pure_dqn_{0}'.format(N))
    for i in range(len(loss_list)):
        writer.add_scalar('loss', loss_list[i], global_step=i)
    print("per_reward_list",per_reward_list)
    print("per_waiting_time_list",per_waiting_time_list)
    print("per_delay_time_list",per_delay_time_list)
    print("per_queue_list",per_queue_list)
    print("loss_list",loss_list)
    print("remain_car_numbers",remain_car_numbers)
    fname = 'data_save/new/data_{0}.xlsx'.format(N)
    record_data(per_reward_list,per_waiting_time_list,per_delay_time_list,per_queue_list,loss_list,remain_car_numbers,fname)