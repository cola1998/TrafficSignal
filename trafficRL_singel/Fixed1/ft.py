import traci
import datetime
import sys
import os
from trafficRL_singel.Fixed1.FixedTime import get_halting_number,get_waiting_times,get_delay_times
def ft_test():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    max_step = 5400
    sumoBinary = 'sumo'  # 'sumo-gui'
    sumoConfig = [sumoBinary, '-c', '../net/netFile/dynamicNet.sumocfg', "--tripinfo-output", "tripinfo.xml"]
    tlsID = 'gneJ0'
    step = 0


    traci.start(sumoConfig)
    tlsID = traci.trafficlight.getIDList()[0]

    traci.trafficlight.setPhase(tlsID, 0)  # 信号灯相位从0开始
    traci.simulationStep()  # 先执行一步
    q = 0
    q2 = 0
    waiting_time = 0
    while traci.simulation.getMinExpectedNumber() > 0 and step < max_step:
        # 仿真环境中有车 并且 step<max_step
        trafficlightIndex = traci.trafficlight.getPhase(tlsID)
        # print(trafficlightIndex) # 显示当前相位
        if trafficlightIndex == 0:  # e
            for i in range(30):
                traci.simulationStep()
            q += get_halting_number(['S', 'N', 'W'])
            q2 += get_halting_number(['N', 'S', 'E', 'W'])
            waiting_time_t = get_waiting_times()
            waiting_time += waiting_time_t
            step += 30

        elif trafficlightIndex == 1:
            for i in range(3):
                traci.simulationStep()
            q += get_halting_number(['S', 'N', 'W'])
            q2 += get_halting_number(['N', 'S', 'E', 'W'])
            waiting_time_t = get_waiting_times()
            waiting_time += waiting_time_t
            step += 3

        elif trafficlightIndex == 2:  # S
            for i in range(30):
                traci.simulationStep()
            q += get_halting_number(['S', 'E', 'W'])
            q2 += get_halting_number(['N', 'S', 'E', 'W'])
            waiting_time_t = get_waiting_times()
            waiting_time += waiting_time_t
            step += 30
        elif trafficlightIndex == 3:  # S
            for i in range(3):
                traci.simulationStep()
            q += get_halting_number(['S', 'E', 'W'])
            q2 += get_halting_number(['N', 'S', 'E', 'W'])
            waiting_time_t = get_waiting_times()
            waiting_time += waiting_time_t
            step += 3

        elif trafficlightIndex == 4:  # E
            for i in range(30):
                traci.simulationStep()
            q += get_halting_number(['S', 'E', 'N'])
            q2 += get_halting_number(['N', 'S', 'E', 'W'])
            waiting_time_t = get_waiting_times()
            waiting_time += waiting_time_t
            step += 30

        elif trafficlightIndex == 5:  # E
            for i in range(3):
                traci.simulationStep()
            q += get_halting_number(['S', 'E', 'N'])
            q2 += get_halting_number(['N', 'S', 'E', 'W'])
            waiting_time_t = get_waiting_times()
            waiting_time += waiting_time_t
            step += 3

        elif trafficlightIndex == 6:  # W
            for i in range(30):
                traci.simulationStep()
            q += get_halting_number(['N', 'E', 'W'])
            q2 += get_halting_number(['N', 'S', 'E', 'W'])
            waiting_time_t = get_waiting_times()
            waiting_time += waiting_time_t
            step += 30

        elif trafficlightIndex == 7:  # W
            for i in range(3):
                traci.simulationStep()
            q += get_halting_number(['N', 'E', 'W'])
            q2 += get_halting_number(['N', 'S', 'E', 'W'])
            waiting_time_t = get_waiting_times()
            waiting_time += waiting_time_t
            step += 3
        elif trafficlightIndex == 8:  # W E
            for i in range(12):
                traci.simulationStep()
            q += get_halting_number(['N', 'S'])
            q2 += get_halting_number(['N', 'S', 'E', 'W'])
            waiting_time_t = get_waiting_times()
            waiting_time += waiting_time_t
            step += 12

        elif trafficlightIndex == 9:  # W
            for i in range(3):
                traci.simulationStep()
            q += get_halting_number(['N', 'S'])
            q2 += get_halting_number(['N', 'S', 'E', 'W'])
            waiting_time_t = get_waiting_times()
            waiting_time += waiting_time_t
            step += 3
        elif trafficlightIndex == 10:  # W E
            for i in range(12):
                traci.simulationStep()
            q += get_halting_number(['W', 'E'])
            q2 += get_halting_number(['N', 'S', 'E', 'W'])
            waiting_time_t = get_waiting_times()
            waiting_time += waiting_time_t
            step += 12
        elif trafficlightIndex == 11:  # W
            for i in range(3):
                traci.simulationStep()
            q += get_halting_number(['W', 'E'])
            q2 += get_halting_number(['N', 'S', 'E', 'W'])
            waiting_time_t = get_waiting_times()
            waiting_time += waiting_time_t
            step += 3
    print("ft traci.simulation.getMinExpectedNumber() and step",traci.simulation.getMinExpectedNumber(),step)
    traci.close()
    return waiting_time

def ft_test2(sumofile):
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    max_step = 7200
    sumoBinary = 'sumo'  # 'sumo-gui'
    sumoConfig = [sumoBinary, '-c', sumofile, "--tripinfo-output", "tripinfo.xml"]
    tlsID = 'gneJ0'
    step = 0


    traci.start(sumoConfig)
    tlsID = traci.trafficlight.getIDList()[0]

    traci.trafficlight.setPhase(tlsID, 0)  # 信号灯相位从0开始
    traci.simulationStep()  # 先执行一步
    q = 0
    q2 = 0
    waiting_time = 0
    waiting_time_l = []
    delay_time = 0
    while traci.simulation.getMinExpectedNumber() > 0 and step<max_step:
        # 仿真环境中有车 并且 step<max_step
        trafficlightIndex = traci.trafficlight.getPhase(tlsID)
        # print(trafficlightIndex) # 显示当前相位
        if trafficlightIndex == 0:  # NS
            for i in range(30):
                traci.simulationStep()
            step += 30
            q += get_halting_number(['E', 'W'])
            q2 += get_halting_number(['N', 'S', 'E', 'W'])
            waiting_time += get_waiting_times()
            delay_time += get_delay_times(step)
        elif trafficlightIndex == 1:
            for i in range(3):
                traci.simulationStep()
            q += get_halting_number(['S', 'N', 'W'])
            q2 += get_halting_number(['N', 'S', 'E', 'W'])
            waiting_time += get_waiting_times()
            step += 3
            delay_time += get_delay_times(step)
        elif trafficlightIndex == 2:  # S
            for i in range(30):
                traci.simulationStep()
            q += get_halting_number(['S', 'E', 'W'])
            q2 += get_halting_number(['N', 'S', 'E', 'W'])
            waiting_time += get_waiting_times()
            step += 30
            delay_time += get_delay_times(step)
        elif trafficlightIndex == 3:  # S
            for i in range(3):
                traci.simulationStep()
            q += get_halting_number(['S', 'E', 'W'])
            q2 += get_halting_number(['N', 'S', 'E', 'W'])
            waiting_time += get_waiting_times()
            step += 3
            delay_time += get_delay_times(step)
        elif trafficlightIndex == 4:  # E
            for i in range(30):
                traci.simulationStep()
            q += get_halting_number(['S', 'E', 'N'])
            q2 += get_halting_number(['N', 'S', 'E', 'W'])
            waiting_time += get_waiting_times()
            step += 30
            delay_time += get_delay_times(step)
        elif trafficlightIndex == 5:  # E
            for i in range(3):
                traci.simulationStep()
            q += get_halting_number(['S', 'E', 'N'])
            q2 += get_halting_number(['N', 'S', 'E', 'W'])
            waiting_time += get_waiting_times()
            step += 3
            delay_time += get_delay_times(step)
        elif trafficlightIndex == 6:  # W
            for i in range(30):
                traci.simulationStep()
            q += get_halting_number(['N', 'E', 'W'])
            q2 += get_halting_number(['N', 'S', 'E', 'W'])
            waiting_time += get_waiting_times()
            step += 30
            delay_time += get_delay_times(step)
        elif trafficlightIndex == 7:  # W
            for i in range(3):
                traci.simulationStep()
            q += get_halting_number(['N', 'E', 'W'])
            q2 += get_halting_number(['N', 'S', 'E', 'W'])
            waiting_time += get_waiting_times()
            step += 3
            delay_time += get_delay_times(step)

    print("ft traci.simulation.getMinExpectedNumber() and step",traci.simulation.getMinExpectedNumber(),step)
    rn = traci.simulation.getMinExpectedNumber()
    traci.close()
    return waiting_time,rn,q2,delay_time

# res = ft_test()
# print(res)