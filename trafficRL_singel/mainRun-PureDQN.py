import os
import sys
import torch
import traci
import datetime
from collections import namedtuple
from trafficRL_singel.dqn_agent import Agent
from trafficRL_singel.env import environment
from trafficRL_singel.trafficSignal import TrafficSignal
from trafficRL_singel.replayBuffer import ReplayBuffer


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    sys.path.append(tools)

if __name__ == "__main__":

    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))  # 命名元组

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
    inEdges = ['gneE0','gneE1','gneE2','gneE3']
    outEdges = ['-gneE0','-gneE1','-gneE2','-gneE3']
    # 缓冲池参数
    capacity = 3000
    minibatch = 100

    # 初始化各个参数
    learning_rate = 0.9  # 学习率
    gamma = 0.9  # 折扣因子
    epsilon = 1  # 探索率
    alpha = 0.2

    action_dim = 8
    state_dim = 9
    mid_dim = 3



    Max_episodes = 10
    Max_steps = 5400
    C = 2
    reward_list = []
    waiting_time_list = []
    sumoBinary = 'sumo-gui'
    sumoConfig = [sumoBinary, '-c', 'net/netFile/lowNet.sumocfg', "--tripinfo-output", "tripinfo.xml"]

    print('----start time: ', datetime.datetime.now())

    # 交通灯参数
    c1 = 0.5
    c2 = 0.1
    c3 = 0.4
    tlsID = 'gneJ0'
    # 初始化交通灯，环境env，智能体agent
    trafficlight = TrafficSignal(tlsID, c1, c2, c3, 30)
    env = environment(inLanes,inEdges, outLanes,outEdges, tlsID, trafficlight)
    buffer = ReplayBuffer(capacity, minibatch)
    agent = Agent(env, state_dim, mid_dim, action_dim, replayBuffer=buffer,
                  alpha=alpha, gamma=gamma, epsilon0=epsilon, learning_rate=learning_rate)



    for episode in range(Max_episodes):

        # 待记录变量
        Q = 0
        Q2 = 0
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
            action = agent.select_action(current_state)
            step,reward = agent.take_action(action,step)
            next_state = agent.get_state()
            buffer.append_buffer(current_state,action,next_state,reward)
            current_state = next_state

            agent.optimize_model()
            R += reward
            waiting_time += env.get_total_waiting_time()
        traci.close()
        reward_list.append(R)
        waiting_time_list.append(waiting_time)
        if episode % C == 0:
            agent.target_net.load_state_dict(agent.net.state_dict())
    print("reward_list : ",reward_list)
    print("waiting_time_list : ",waiting_time_list)