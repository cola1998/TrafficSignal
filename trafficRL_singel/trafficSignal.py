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
        self.cumulate_unselect_steps = [0 for i in range(8)]
        # 搞一个列表 记录每个相位累计未被选择时间
        # 选择了就清零
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    def getPhase(self):
        # 传入tlsID  根据交通灯的ID返回当前相位的索引
        return traci.trafficlight.getPhase(self.id)
    def compute_unselect_time(self,select_id,time):
        for i in range(len(self.cumulate_unselect_steps)):
            if i != select_id:
                self.cumulate_unselect_steps[i] += time
    def update(self):
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            traci.trafficlight.setPhase(self.id, self.nextPhase)
            self.is_yellow = False

    def setNextPhase(self, newPhaseIndex):  # 如何切换到下一个状态
        nowPhase = traci.trafficlight.getPhase(self.id)
        # 如果当前相位和下一个相位一致  直接切换
        if self.nowPhase == newPhaseIndex or self.time_since_last_phase_change < self.yellow_time + self.green_time:
            traci.trafficlight.setPhase(self.id, newPhaseIndex)
            self.next_action_time = self.env
        else:
            # 过渡到黄灯 然后再执行下一个相位
            traci.trafficlight.setPhase(self.id, self.nowPhase + 1)  # 变成黄灯
            self.nextPhase = 0
            self.is_yellow = True
            self.time_since_last_phase_change = 0

    def compute_reward(self, t,inLanesTotal):  # 传入当前时间步 和 free_speed
        delay_time = 0  # 延迟时间
        for lane in inLanesTotal:
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
        p_count = sum(self.cumulate_unselect_steps)

        identify_reward = 0
        # 计算不平衡车流的计算公式待查找！！
        '''
        先计算交通量方向分布系数
        '''

        return self.c1 * delay_time + self.c2 * p_count + self.c3 * identify_reward