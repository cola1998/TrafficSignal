import os
import sys
import torch
import traci
import datetime
from collections import namedtuple
from trafficRL_singel.BIUD3QN_Algorithm.d3qn_agent import Agent
from trafficRL_singel.BIUD3QN_Algorithm.env import environment
from trafficRL_singel.BIUD3QN_Algorithm.trafficSignal import TrafficSignal
from trafficRL_singel.BIUD3QN_Algorithm.replayBuffer import ReplayBuffer
from trafficRL_singel.DQN.recordData import record_data,record_data_ft
from tensorboardX import SummaryWriter
from trafficRL_singel.Fixed1.ft import ft_test2
from trafficRL_singel.DQN.unbalance_net.generateCarFlow import generate_routefile
from trafficRL_singel.DQN.get_car_info import get_car_number_4

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)

if __name__ == "__main__":
    # N = '29_high_改变reward'
    N = '27_high_un'

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

    action_dim = 4
    state_dim = 9
    mid_dim = 3

    Max_episodes = 100
    Max_steps = 7200
    C = 10

    sumoBinary = 'sumo'  # 'sumo-gui'
    sumofile = '../DQN/unbalance_net/net.sumocfg'
    sumoConfig = [sumoBinary, '-c', sumofile, "--tripinfo-output", "tripinfo.xml"]

    print('----start time: ', datetime.datetime.now())

    # 交通灯参数
    c1 = 0
    c2 = 0.5
    c3 = 0.4
    tlsID = 'gneJ0'

    # 初始化交通灯，环境env，智能体agent
    trafficlight = TrafficSignal(tlsID, c1, c2, c3, 30)
    env = environment(inLanes, inEdges, outLanes, outEdges, tlsID, trafficlight)
    buffer = ReplayBuffer(capacity, minibatch)
    agent = Agent(env, state_dim, mid_dim, action_dim, replayBuffer=buffer,
                  alpha=alpha, gamma=gamma, epsilon0=epsilon, learning_rate=learning_rate)
    dummy_input = torch.rand(1,state_dim)
    with SummaryWriter('runs/graph/{0}/'.format(N)) as w:
        w.add_graph(agent.cri,dummy_input)

    # 一些关于车流的参数
    car_N = 3000 # 3000
    # 不同的转弯比例
    per_reward_list = []
    per_waiting_time_list = []
    per_delay_time_list = []
    per_queue_list = []
    per_queue_list2 = []
    loss_list = []
    d3qn_throughput = []
    per_loss_list = []

    ft_waiting_time_list = []
    ft_delay_time_list = []
    ft_queue_list = []
    ft_throughput = []
    net_update_times = 0


    sumofile_name = '../DQN/unbalance_net/net_3000.rou.xml'
    path2 = '../DQN/unbalance_net/net_3000.rou.xml'
    # 不同的转弯比例
    L = 6
    sRatio = 3
    cz = [] # 记录fixed 与 d3qn的waiting time差值

    record = '''        
                数据：{0}
                奖励设置：-c1*delay_time - c2*per_waiting_time
                奖励参数调整：c1={1} c2={2}
                参数设置:1) 探索率 {3} 动态改变
                2) 学习率 {4}
                未增加k机制
                optimizer Adam
                查看perloss的变化
                转弯比{5}
                '''.format(sumofile, c1, c2, epsilon, learning_rate, L)
    with open('record.txt', 'a+') as f:
        f.writelines("文件名:" + N)
        f.writelines(record)

    for episode in range(Max_episodes):
        # 每一轮都要重新生成车流文件,避免过拟合现象
        generate_routefile(sumofile_name, car_N,Max_steps , L, sRatio)
        get_car_number_4(path2,episode)
        ft_waiting_time,rn,ft_queue,ft_delay_time = ft_test2(sumofile)
        ft_waiting_time_list.append(round(ft_waiting_time/car_N,2))
        ft_delay_time_list.append(round(ft_delay_time/car_N,2))
        ft_queue_list.append(round(ft_queue/car_N,2))
        ft_throughput.append(car_N-rn)
        print("fixed time {0} finished".format(episode))
        # 待记录变量
        R = 0
        waiting_time = 0
        queue = 0
        delay_time = 0
        total_loss = 0

        traci.start(sumoConfig)
        tlsID = traci.trafficlight.getIDList()[0]
        traci.trafficlight.setPhase(tlsID, 0)  # 信号灯相位从0开始
        traci.simulationStep()
        current_state = agent.get_state()

        step = 0
        while traci.simulation.getMinExpectedNumber() > 0 and step < Max_steps:  #
            res = agent.identifyFlow()
            if res != None:
                flowTag = res[0]  # 'k' and 'ra'
                flowIndex = int(res[1])  # 0,1,2,3
            else:  # 表明是平衡车流
                flowTag = 'ba'
                flowIndex = 0
            current_state = torch.tensor(current_state, dtype=torch.float32)
            current_state = current_state.view(1, state_dim)
            action = agent.select_action(current_state,epsilon0=round(episode / Max_episodes, 2))  #
            step, reward,waiting_time_t,delay_time1,queue1 = agent.take_action(action, step, flowTag, flowIndex)
            next_state = agent.get_state()
            if step >= Max_steps:
                buffer.append_buffer(current_state, action, next_state, reward, 0.0)
            else:
                buffer.append_buffer(current_state, action, next_state, reward, gamma)
            current_state = next_state

            loss = agent.optimize_model()

            if loss != None:
                # print("main loss", loss)
                loss_list.append(loss)
                total_loss += loss
            R += reward
            waiting_time += waiting_time_t
            delay_time += delay_time1
            queue += queue1
            net_update_times += 1
            if net_update_times % C == 0:
                agent.optimize_target_model()

        print("episode= {0} 结束时traci.simulation.getMinExpectedNumber={1} 和 step={2}".format(episode,
                                                                                            traci.simulation.getMinExpectedNumber(),
                                                                                            step))

        per_waiting_time_list.append(round(waiting_time/car_N,2))
        per_reward_list.append(round(R/net_update_times,2))
        per_delay_time_list.append(round(delay_time / car_N, 2))
        per_queue_list.append(round(queue / car_N, 2))
        d3qn_throughput.append(car_N-traci.simulation.getMinExpectedNumber())
        per_loss_list.append(round(total_loss/net_update_times,2))
        cz.append(ft_waiting_time_list[len(ft_waiting_time_list)-1]-per_waiting_time_list[len(per_waiting_time_list)-1])

        traci.close()
    os.mkdir('policy_save/new_runs/{0}'.format(N))
    writer = SummaryWriter('policy_save/new_runs/{0}/biu_d3qn_{0}'.format(N))
    for i in range(len(per_waiting_time_list)):
        writer.add_scalar('waiting_time', per_waiting_time_list[i], global_step=i)
        writer.add_scalar('reward', per_reward_list[i], global_step=i)
        writer.add_scalar('delay_time', per_delay_time_list[i], global_step=i)
        writer.add_scalar('queue', per_queue_list[i], global_step=i)
        writer.add_scalar('throughput',d3qn_throughput[i],global_step=i)
        writer.add_scalar('per_loss',per_loss_list[i],global_step=i)
        writer.add_scalar('cz', cz[i], global_step=i)
    writer = SummaryWriter('policy_save/new_runs/{0}/loss_biu_d3qn_{0}'.format(N))
    for i in range(len(loss_list)):
        writer.add_scalar('loss', loss_list[i], global_step=i)
    writer = SummaryWriter('policy_save/new_runs/{0}/ft_example_{0}'.format(N))
    for i in range(len(ft_waiting_time_list)):
        writer.add_scalar('waiting_time', ft_waiting_time_list[i], global_step=i)
        writer.add_scalar('delay_time', ft_delay_time_list[i], global_step=i)
        writer.add_scalar('queue', ft_queue_list[i], global_step=i)
        writer.add_scalar('throughput',ft_throughput[i],global_step=i)
    fname = 'data_save/new/data_{0}.xlsx'.format(N)
    record_data(per_reward_list, per_waiting_time_list, per_delay_time_list, per_queue_list, loss_list,
                d3qn_throughput, fname)
    fname2 = 'data_save/new/ft_data_{0}.xlsx'.format(N)
    record_data_ft(ft_waiting_time_list,ft_delay_time_list,ft_queue_list,ft_throughput,fname2)
