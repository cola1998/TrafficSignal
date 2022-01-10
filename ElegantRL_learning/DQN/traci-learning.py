import traci
import datetime
import sys
import os
'''
启动traci 并给定一些参数
sumo-gui 表示启动sumo-gui界面
cfg文件是sumo中的net，rou 路网车辆信息的文件

'''
# 检查系统路径
if 'SUMO_HOME' in os.environ:
     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
     sys.path.append(tools)
else:
     sys.exit("please declare environment variable 'SUMO_HOME'")


sumoBinary = 'sumo-gui'  # '--nogui'

traci.start([sumoBinary, '-c', 'netFile/net1.sumocfg', "--tripinfo-output", "tripinfo.xml",'--duration-log.statistics'])
print('----start time: ', datetime.datetime.now())

def run():
    step = 0
    # traci.trafficlight.setPhase("gneJ0", 2)   #0 是指traffic light 的id，四个phase依次编号为0，1，2，3，初始时设置phase为2
    while traci.simulation.getMinExpectedNumber() > 0:
        # 该函数得到当前net中车辆数目加上还没有进入net的车辆数目，只要大于0就表示还有车辆需要处理
        traci.simulationStep()  # 运行一步仿真
        # if traci.trafficlight.getPhase("gneJ0") == 2:
        #     if traci.inductionloop.getLastStepVehicleNumber("gneJ0") > 0:# 在上一步仿真中，经过induction loop的车数
        #         traci.trafficlight.setPhase("gneJ0",3) # 如果有车进来，则切换phase
        #     else:
        #         traci.trafficlight.setPhase("gneJ0",2) # 否则依然保持phase。注意，这里重置了phase，所以会重新计时。
        # step += 1
    traci.close()
    sys.exit()
run()
