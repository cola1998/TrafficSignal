import numpy as np
import pandas as pd
import traci


def build_env():
    '''
    初始化一个环境
    :return:
    '''
    pass





class environment():
    def __init__(self, inLanes,inEdges, outLanes,outEdges,tlsID,trafficlight):
        self.inLanes = inLanes
        self.inEdges = inEdges
        self.outLanes = outLanes
        self.outEdges = outEdges
        self.free_speed = 13.58  # 每个环境中有一个自己的free_speed吧
        self.tlsID = tlsID
        self.trafficlight = trafficlight  # 环境中有一个交通灯对象

    def reset(self):
        '''
        重启 初始化一个环境
        :return:
        '''
        pass



    def step(self, current_action,step):  # 传入的是下一个要执行的相位
        '''
        执行一步
        :return:
        '''
        last_action = self.trafficlight.getPhase()
        # current_action
        # 可选action的list [0,1,2,3,4,5,6,7]

        if last_action != current_action:  # 需要切换
            current_action = current_action * 2
            traci.trafficlight.setPhase(self.tlsID, last_action + 1)
            for i in range(3):
                traci.simulationStep()
            self.trafficlight.compute_unselect_time(last_action,3)

            traci.trafficlight.setPhase(self.tlsID, current_action)
            for i in range(30):
                traci.simulationStep()
            self.trafficlight.compute_unselect_time(current_action//2, 30)
            step += 33
        else:  # 不需要切换
            for i in range(30):
                traci.simulationStep()
            self.trafficlight.compute_unselect_time(current_action, 30)
            step += 30
        return step

    def get_state(self):
        # 等待队列的长度
        # s = []
        # for lane in self.inLanes:
        #     s.append(traci.lane.getLastStepHaltingNumber(lane))
        # return s
        # 9维状态
        # 4维 每条路排队长度  4维 每条路转向比  1维当前相位
        s = []
        for edge in self.inEdges:
            s.append(traci.edge.getLastStepHaltingNumber(edge))
        ls,ss,rs = self.get_turn_number_edge()
        for i in range(len(ls)):
            if (ls[i] + ss[i] + rs[i]) != 0:
                s.append(round(ls[i]/ (ls[i] + ss[i] + rs[i]),2))
            else:
                s.append(0)
        s.append(self.trafficlight.getPhase()//2)
        return s

    def get_lane_density(self):
        vehicle_size_min_gap = 7.5  # 5veh_size 2.5gap
        out_lanes_density = []
        for lane in self.outLanes:
            density = traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)
            out_lanes_density.append(min(1, density))
        return out_lanes_density

    def get_lanes_queue(self, current_lane):  # 获取当前车道的排队车辆长度
        return traci.lane.getLastStepHaltingNumber(current_lane)

    def get_total_lanes_queue(self):  # lanes 是所有入口车道的lane id
        res = 0
        for lane in self.inLanes:
            res += traci.lane.getLastStepHaltingNumber(lane)
        return res

    def get_total_waiting_time(self):
        total_waiting_time = 0
        for line in self.inLanes:
            veh_list = traci.lane.getLastStepVehicleIDs(line)
            waiting_time = 0
            for veh in veh_list:
                waiting_time += traci.vehicle.getAccumulatedWaitingTime(veh)
            total_waiting_time += waiting_time
        return total_waiting_time

    # def get_waiting_time_per_line(self):
    #     wait_time_per_lane = []
    #     for line in self.inLanes:  # lane是lane_id
    #         veh_list = traci.lane.getLastStepVehicleIDs(line)  # 返回上一时间步在给定车道lane的车辆id列表
    #         wait_time = 0
    #         for veh in veh_list:
    #             veh_lane = traci.vehicle.getLaneID(veh)  # 获取车辆veh的车道
    #             acc = traci.vehicle.getAccumulatedWaitingTime(veh)  # 返回给定车辆的累计等待时间
    #             if veh not in env.vehicles:  # ??? 不在环境中 将其加入
    #                 env.vehicles[veh] = {veh_lane: acc}
    #             else:  # 遍历环境中这辆车在的每一条车道，如果不是当前道路就累计时间
    #                 env.vehicles[veh][veh_lane] = acc - sum(
    #                     [env.vehicles[veh][veh_lane] for lane in env.vehicles[veh].keys() if lane != veh_lane])
    #             wait_time += env.vehicles[veh][veh_lane]
    #         wait_time_per_lane.append(wait_time)
    #     return wait_time_per_lane
    def get_turn_number_edge(self):
        s_counts, l_counts, r_counts = [], [], []
        for edge in self.inEdges:
            veh_list = traci.edge.getLastStepVehicleIDs(edge)
            s_count = 0  # 直行比例
            l_count = 0  # 左转比例
            r_count = 0  # 右转比例
            for veh in veh_list:
                now_edge = traci.vehicle.getRoadID(veh)
                route = list(traci.vehicle.getRoute(veh))
                next_edge = route[route.index(now_edge) + 1]
                # 左转
                if veh[:3] == 'N_E' or veh[:3] == 'S_W' or veh[:3] == 'E_N' or veh[:3] == 'W_S':
                    l_count += 1
                # 直行
                elif veh[:3] == 'N_S' or veh[:3] == 'S_N' or veh[:3] == 'W_E' or veh[:3] == 'E_W':
                    s_count += 1
                else:
                    r_count += 1
            l_counts.append(l_count)
            s_counts.append(s_count)
            r_counts.append(r_count)
        return s_counts, l_counts, r_counts

    def get_turn_number_lane(self):
        l_counts, s_counts, r_counts = [],[],[]
        # 可否获取车辆再某条道路左转的比例以及直行的比例 还有右转
        for lane in self.inLanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            s_count = 0 # 直行比例
            l_count = 0  # 左转比例
            r_count = 0 # 右转比例
            for veh in veh_list:
                now_edge = traci.vehicle.getRoadID(veh)
                route = list(traci.vehicle.getRoute(veh))
                next_edge = route[route.index(now_edge) + 1]
                # 左转
                if veh[:3] == 'N_E' or veh[:3] == 'S_W' or veh[:3] == 'E_N' or veh[:3] == 'W_S':
                    l_count += 1
                # 直行
                elif veh[:3] == 'N_S' or veh[:3] == 'S_N' or veh[:3] == 'W_E' or veh[:3] == 'E_W':
                    s_count += 1
                else:
                    r_count += 1
            l_counts.append(l_count)
            s_counts.append(s_count)
            r_counts.append(r_count)
        return l_counts,s_counts,r_counts


    def get_reward(self,step):
        reward = self.trafficlight.compute_reward(step,self.free_speed,self.inLanes)
        return reward
