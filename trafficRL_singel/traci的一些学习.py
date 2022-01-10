'''
状态：等待队列的长度
奖励：
（1）延迟时间
（2）相位长时间未被选择
（3）识别不平衡车流

sumo中的时间问题
depart 出发时间

traci的一些接口
lane 包
traci.lane.getLastStepHaltingNumber(laneID)  上一时间步给定车道的停止车辆的数量
traci.lane.getLastStepVehicleIDs(laneID)   上一时间步给定车道上车辆id
traci.lane.getLastStepVehicleNumber(laneID)  上一时间步给定车道的车辆数

traci.lane.getLength(laneID)  给定车道id的车道长度
traci.lane.getTravelTime(laneID) ??

vehicle包
traci.vehicle.getAccumulatedWaitingTime(vehID)  车辆在一定时间范围内的累计等待时间
        600(m)/13.89(m/s) = 43秒  所以默认的100s足够计算车辆进入路网后整个的等待时间吧
        '--waiting-time-memory'  修改的参数
traci.vehicle.getDistance(vehID) 车辆到起点的距离
traci.vehicle.getLaneID(vehID)    返回车辆上一时间步所在车道的id

trafficlight包
traci.trafficlight.getPhase()  返回当前相位的索引
traci.trafficlight.getPhaseDuration()  返回当前相位的总持续时间
traci.trafficlight.getPhaseName()
traci.trafficlight.setPhase()  设置下一阶段的相位
traci.trafficlight.getNextSwitch()  获取到下个信号灯相位的时间 return double

simulation包
traci.simulation.getCurrentTime()  获取当前仿真时间 ？？
traci.simulationStep() 执行一步

edge包

'''

import traci


def get_lane_density(out_lanes):
    vehicle_size_min_gap = 7.5  # 5veh_size 2.5gap
    out_lanes_density = []
    for lane in out_lanes:
        density = traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)
        out_lanes_density.append(min(1, density))
    return out_lanes_density


def get_lanes_queue(current_lane):  # 获取当前车道的排队车辆长度
    return traci.lane.getLastStepHaltingNumber(current_lane)


def get_total_lanes_queue(inLanes):  # lanes 是所有入口车道的lane id
    res = 0
    for lane in inLanes:
        res += traci.lane.getLastStepHaltingNumber(lane)
    return res


def get_waiting_time_per_line(inLanes):
    wait_time_per_lane = []
    for line in inLanes:  # lane是lane_id
        veh_list = traci.lane.getLastStepVehicleIDs(line)  # 返回上一时间步在给定车道lane的车辆id列表
        wait_time = 0
        for veh in veh_list:
            veh_lane = traci.vehicle.getLaneID(veh)  # 获取车辆veh的车道
            acc = traci.vehicle.getAccumulatedWaitingTime(veh)  # 返回给定车辆的累计等待时间
            if veh not in env.vehicles:  # ??? 不在环境中 将其加入
                env.vehicles[veh] = {veh_lane: acc}
            else:  # 遍历环境中这辆车在的每一条车道，如果不是当前道路就累计时间
                env.vehicles[veh][veh_lane] = acc - sum(
                    [env.vehicles[veh][veh_lane] for lane in env.vehicles[veh].keys() if lane != veh_lane])
            wait_time += env.vehicles[veh][veh_lane]
        wait_time_per_lane.append(wait_time)
    return wait_time_per_lane

