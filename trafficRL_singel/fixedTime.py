import traci
import os
import sys
import datetime

if __name__ == '__main__':
    # 开启sumo仿真
    # 检查sumo的配置
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'],'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    sumoBinary = 'sumo-gui'  # '--nogui'

    traci.start([sumoBinary, '-c', 'netFile/net1.sumocfg', "--tripinfo-output", "tripinfo.xml"])
    print('----start time: ', datetime.datetime.now())

    while traci.simulation.getMinExpectedNumber() > 0:  # 系统中仿真的车辆数
        traci.simulationStep()  # 运行一步仿真

    traci.close()
    sys.exit()