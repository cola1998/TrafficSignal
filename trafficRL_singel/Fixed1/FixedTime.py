import traci
import datetime
import sys
import os
'''

fixedTime  main  sorry!!!
启动traci 并给定一些参数
sumo-gui 表示启动sumo-gui界面
cfg文件是sumo中的net，rou 路网车辆信息的文件
'''

def get_halting_number(query):
    res = 0
    for i in range(len(query)):
        if query[i] == 'N':
            res += traci.edge.getLastStepVehicleNumber('gneE1')  # N
        elif query[i] == 'S':
            res += traci.edge.getLastStepVehicleNumber('gneE3')  # S
        elif query[i] == 'W':
            res += traci.edge.getLastStepVehicleNumber('gneE2') # W
        else:
            res += traci.edge.getLastStepVehicleNumber('gneE0')
    return res

def get_waiting_times():
    inEdges = ['gneE0', 'gneE1', 'gneE2', 'gneE3']
    waiting_times = 0
    waiting_times2 = 0
    # 试一下两种方式获取的waiting_time是否一致
    for edge in inEdges:
        veh_list = traci.edge.getLastStepVehicleIDs(edge)
        for veh in veh_list:
            waiting_times += traci.vehicle.getAccumulatedWaitingTime(veh)
    # print("方式一计算waiting time：", waiting_times)
    # for veh in traci.vehicle.getIDList():
    #     # traci.vehicle.getRoadID()
    #     if traci.vehicle.getRoadID(veh) in inEdges:
    #         waiting_times2 += traci.vehicle.getAccumulatedWaitingTime(veh)
    # print("方式二计算waiting time：", waiting_times2)
    return waiting_times

def get_delay_times(t):
    inLanes = ['gneE0_0', 'gneE0_1', 'gneE0_2', 'gneE0_3',
        'gneE1_0', 'gneE1_1', 'gneE1_2', 'gneE1_3',
        'gneE2_0', 'gneE2_1', 'gneE2_2', 'gneE2_3',
        'gneE3_0', 'gneE3_1', 'gneE3_2', 'gneE3_3']
    delay_time = 0
    for lane in inLanes:
        free_speed = traci.lane.getMaxSpeed(lane)
        for veh in traci.lane.getLastStepVehicleIDs(lane):
            speed = traci.vehicle.getSpeed(veh)
            d = 1 - (speed/free_speed)
            delay_time += d
    return delay_time

# # 检查系统路径
# if 'SUMO_HOME' in os.environ:
#      tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#      sys.path.append(tools)
# else:
#      sys.exit("please declare environment variable 'SUMO_HOME'")



# traci.start()
# print('----start time: ', datetime.datetime.now())
# step = 0 # 一个step1000ms左右？？？迷惑
# while traci.simulation.getMinExpectedNumber() > 0:
    # traci.simulationStep()
    # step += 1
    # print(step/10)
    # traci.simulation.getCurrentTime()/1000  这样能和进入仿真环境时间匹配上

# max_episodes = 1 # 训练次数
# max_step = 5400
# waiting_time_list = []
# queue_list = []
# queue_list2 = []
# sumoBinary = 'sumo'  # 'sumo-gui'
# sumoConfig = [sumoBinary, '-c', 'netFile1/net1.sumocfg', "--tripinfo-output", "tripinfo.xml"]
# tlsID = 'gneJ0'
# step = 0
#
# for episode in range(max_episodes):
#     traci.start(sumoConfig)
#     tlsID = traci.trafficlight.getIDList()[0]
#     # print(tlsID)
#
#     traci.trafficlight.setPhase(tlsID,0)  # 信号灯相位从0开始
#     traci.simulationStep() # 先执行一步
#     q = 0
#     q2 = 0
#     waiting_time = 0
#     while traci.simulation.getMinExpectedNumber() > 0 and step < max_step:
#         # 仿真环境中有车 并且 step<max_step
#         trafficlightIndex = traci.trafficlight.getPhase(tlsID)
#         # print(trafficlightIndex) # 显示当前相位
#         if trafficlightIndex == 0: # N
#             for i in range(30):
#                 traci.simulationStep()
#             q += get_halting_number(['S', 'E', 'W'])
#             q2 += get_halting_number(['N', 'S', 'E', 'W'])
#             waiting_time += get_waiting_times()
#         elif trafficlightIndex == 1:
#             for i in range(3):
#                 traci.simulationStep()
#             q += get_halting_number(['S', 'E', 'W'])
#             q2 += get_halting_number(['N', 'S', 'E', 'W'])
#             waiting_time += get_waiting_times()
#         elif trafficlightIndex == 2: # S
#             for i in range(30):
#                 traci.simulationStep()
#             q += get_halting_number(['N', 'E', 'W'])
#             q2 += get_halting_number(['N', 'S', 'E', 'W'])
#             waiting_time += get_waiting_times()
#         elif trafficlightIndex == 3: # S
#             for i in range(3):
#                 traci.simulationStep()
#             q += get_halting_number(['N', 'E', 'W'])
#             q2 += get_halting_number(['N', 'S', 'E', 'W'])
#             waiting_time += get_waiting_times()
#         elif trafficlightIndex == 4:  # E
#             for i in range(30):
#                 traci.simulationStep()
#             q += get_halting_number(['S', 'N', 'W'])
#             q2 += get_halting_number(['N', 'S', 'E', 'W'])
#             waiting_time += get_waiting_times()
#         elif trafficlightIndex == 5:  # E
#             for i in range(3):
#                 traci.simulationStep()
#             q += get_halting_number(['S', 'N', 'W'])
#             q2 += get_halting_number(['N', 'S', 'E', 'W'])
#             waiting_time += get_waiting_times()
#         elif trafficlightIndex == 6:  # W
#             for i in range(30):
#                 traci.simulationStep()
#             q += get_halting_number(['S', 'E', 'N'])
#             q2 += get_halting_number(['N', 'S', 'E', 'W'])
#             waiting_time += get_waiting_times()
#         elif trafficlightIndex == 7:  # W
#             for i in range(3):
#                 traci.simulationStep()
#             q += get_halting_number(['S', 'E', 'N'])
#             q2 += get_halting_number(['N', 'S', 'E', 'W'])
#             waiting_time += get_waiting_times()
#     queue_list.append(q)
#     queue_list2.append(q2)
#     waiting_time_list.append(waiting_time)
#     traci.close()
# print("queue_list: ",queue_list)
# print("queue_list2: ",queue_list2)
# print("waiting_time_list: ",waiting_time_list)
# plot_Fixed(max_episodes,queue_list,waiting_time_list)
# sys.exit()