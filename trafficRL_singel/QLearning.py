import sys
import os
import traci
import datetime
from trafficRL_singel.env import environment
from trafficRL_singel.qlearning_agent import Agent
from trafficRL_singel.trafficSignal import TrafficSignal
from trafficRL_singel.replayBuffer import ReplayBuffer

# 检查系统路径
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

max_episodes = 2  # 训练次数
max_step = 5400
waiting_time_list = []
queue_list = []
queue_list2 = []
sumoBinary = 'sumo-gui'  # 'sumo-gui'
module_path = os.path.dirname(__file__)
sumoConfig = [sumoBinary, '-c', 'net/netFile/net0.sumocfg', "--tripinfo-output", "tripinfo.xml"]
tlsID = 'gneJ0'
step = 0
print('----start time: ', datetime.datetime.now())
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
c1 = 0
c2 = 0
c3 = 0
learning_rate = 0.9
epsilon = 0.01  # 后续待动态改变 暂时先这样
for episode in range(max_episodes):
    Q = 0
    Q2 = 0
    waiting_time = 0
    R = 0

    traci.start(sumoConfig)
    tlsID = traci.trafficlight.getIDList()[0]
    traci.trafficlight.setPhase(tlsID, 0)  # 信号灯相位从0开始
    trafficlight = TrafficSignal(tlsID, c1, c2, c3, 30)
    env = environment(inLanes, outLanes, tlsID, trafficlight)  # 先初始化一个环境
    # 也可以直接从环境中获取吧？？
    action_dim = 8
    state_dim = 1  # ??
    mid_dim = 4
    M = 200  # memory 荣量
    minibatch = 50  # 每次采样大小最少m条才能开始训练
    # replayBuffer = ReplayBuffer(M, minibatch)
    agent = Agent(learning_rate, env, action_dim, state_dim, mid_dim)  # 初始化一个agent  利用agent与环境交互 以及memory buffer
    traci.simulationStep()  # 先执行一步
    current_state = agent.get_state()
    while traci.simulation.getMinExpectedNumber() > 0 and step < max_step:
        action = agent.choose_action(current_state)
        # 执行动作
        step = agent.take_action(action, step)

        # if agent.replayBuffer.get_size() >= minibatch:
        #     agent.train()
        last_state = current_state
        current_state,reward = agent.update(step)

        traj = [last_state, action, reward, current_state]
        # agent.replayBuffer.append_buffer(traj)
