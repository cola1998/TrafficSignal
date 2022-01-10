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
��Ҫ�洢������
reward delay_time waiting_time queue throughOutput total_loss loss_list
'''
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)

if __name__ == '__main__':
    N = ""

    # ��ʼ��
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    # ��������
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

    # ����ز���
    capacity = 1000
    minibatch = 32

    # ���������
    learning_rate = 0.005  # ѧϰ��
    gamma = 0.9  # �ۿ�����
    epsilon = 0.5  # ̽����
    alpha = 0.3

    # ���������
    action_dim = 4
    state_dim = 9
    mid_dim = 3

    Max_episodes = 100
    Max_steps = 7200
    C = 10

    # sumo ����
    sumoBinary = 'sumo'  # 'sumo-gui'
    sumofile = '../DQN/unbalance_net/net.sumocfg'
    sumoConfig = [sumoBinary, '-c', sumofile, "--tripinfo-output", "tripinfo.xml"]

    print('----start time: ', datetime.datetime.now())

    # ���㽱������
    biu_c1 = 0
    biu_c2 = 0.5
    biu_c3 = 0.4

    # ��ʼ��BIU D3QN��ͨ�ƣ�����env��������agent
    biu_tlsID = 'gneJ0'
    biu_trafficlight = biu_TrafficSignal(biu_tlsID, biu_c1, biu_c2, biu_c3, 30)
    biu_env = biu_environment(inLanes, inEdges, outLanes, outEdges, biu_tlsID, biu_trafficlight)
    biu_buffer = biu_ReplayBuffer(capacity, minibatch)
    biu_agent = biu_Agent(biu_env, state_dim, mid_dim, action_dim, replayBuffer=biu_buffer,
                  alpha=alpha, gamma=gamma, epsilon0=epsilon, learning_rate=learning_rate)

    dqn_c1 =
    dqn_c2 =
    dqn_c3 =
    # ��ʼ��BIU D3QN��ͨ�ƣ�����env��������agent
    dqn_tlsID = 'gneJ0'
    dqn_trafficlight = dqn_TrafficSignal(dqn_tlsID, dqn_c1, dqn_c2, dqn_c3, 30)
    dqn_env = dqn_environment(inLanes, inEdges, outLanes, outEdges, dqn_tlsID, dqn_trafficlight)
    dqn_buffer = dqn_ReplayBuffer(capacity, minibatch)
    dqn_agent = dqn_Agent(dqn_env, state_dim, mid_dim, action_dim, replayBuffer=dqn_buffer,
                  alpha=alpha, gamma=gamma, epsilon0=epsilon, learning_rate=learning_rate)

    # �����ļ������Ϣ �������� ת��ȵ�
    car_N = 3000  # 3000
    sumofile_name = '../DQN/unbalance_net/net_3000.rou.xml'
    path2 = '../DQN/unbalance_net/net_3000.rou.xml'
    L = 6 # ��ת����
    sRatio = 3 # ֱ�б���

    # ������¼
    record = '''        
    ���ݣ�{0}
    �������ã�-c1*delay_time - c2*per_waiting_time
    ��������������c1={1} c2={2}
    ��������:1) ̽���� {3} ��̬�ı�
            2) ѧϰ�� {4}
            δ����k����
            �Ż�����optimizer Adam
            �鿴perloss�ı仯
            ת���{5}
    '''.format(sumofile, c1, c2, epsilon, learning_rate, L)
    with open('record.txt', 'a+') as f:
        f.writelines("�ļ���:" + N)
        f.writelines(record)

    # ����¼����
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

    # ��ʼ ѵ��
    for episode in range(Max_episodes):
        main_biud3qn()
        main_dqn()