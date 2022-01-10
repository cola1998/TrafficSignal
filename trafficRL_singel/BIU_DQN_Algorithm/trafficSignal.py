import traci

tsl_id = 'gneJ0'
phaseIndex = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


class TrafficSignal:
    def __init__(self, ts_id, c1, c2, c3, green_time):
        self.id = ts_id
        self.nowPhase = 0
        self.nextPhase = -1
        self.time_since_last_phase_change = 0
        self.last_reward = 0
        self.is_yellow = False  # 标记当前是否为黄灯
        self.yellow_time = 3  # 单位s
        self.green_time = green_time

        # 选择了就清零
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    def getPhase(self):
        # 传入tlsID  根据交通灯的ID返回当前相位的索引
        return traci.trafficlight.getPhase(self.id)


    def update(self):
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            traci.trafficlight.setPhase(self.id, self.nextPhase)
            self.is_yellow = False


    def compute_reward(self, t,inLanes):  # 传入当前时间步 和 free_speed
        delay_time = 0  # 延迟时间
        n = 0
        for lane in inLanes:
            free_speed = traci.lane.getMaxSpeed(lane)
            n += len(traci.lane.getLastStepVehicleIDs(lane))
            for veh in traci.lane.getLastStepVehicleIDs(lane):
                edgeID = traci.lane.getEdgeID(lane)
                pos = traci.vehicle.getLanePosition(veh)
                dn = traci.vehicle.getDrivingDistance(veh, edgeID, pos)
                da = traci.vehicle.getLength(veh)  # 是否是该车辆的全程路长？？
                ds = da - dn
                tr = ds / free_speed
                tf = da / free_speed
                d = (t + tr) / tf
                delay_time += d
        if n == 0:
            return 0
        per_delay_time = round(delay_time / n, 2)

        total_waiting_time = 0
        n = 0
        for line in inLanes:
            veh_list = traci.lane.getLastStepVehicleIDs(line)
            waiting_time = 0
            n += len(veh_list)
            for veh in veh_list:
                waiting_time += traci.vehicle.getAccumulatedWaitingTime(veh)
            total_waiting_time += waiting_time
        if n == 0:
            return 0
        per_waiting_time = round(total_waiting_time / n, 2)
        return -self.c1 * per_delay_time - self.c2 * per_waiting_time