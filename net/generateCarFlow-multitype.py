import numpy as np
import math


def generate_routefile(file_name,N, maxStep):
    # 汽车的生成根据weibull分布 生成N辆车
    timings = np.random.weibull(2, N)  # N辆车产生的时间按照weibull分布
    timings = np.sort(timings)

    print("timings", len(timings))
    # 重新调整 以适应0：maxStep的间隔
    car_gen_steps = []
    min_old = math.floor(timings[1])
    max_old = math.ceil(timings[-1])
    print(min_old)
    print(max_old)
    min_new = 0
    max_new = maxStep

    for value in timings:
        tmp = ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new

        car_gen_steps = np.append(car_gen_steps, tmp)
    car_gen_steps = np.rint(car_gen_steps)  # 将数组的元素四舍五入到最接近的整数
    print(car_gen_steps)
    # 生成汽车路径文件 每行一辆车
    with open(file_name, 'w') as routes:
        print("""<routes>
        <vType id="s_car" length="3.0" maxSpeed="15" minGap="1.0" accel="1.0" decel="4.0" />  # 直行车辆
        <vType id="l_car" length="3.0" maxSpeed="15" minGap="1.0" accel="1.0" decel="4.0" />  # 左转车辆
        <vType id="r_car" length="3.0" maxSpeed="15" minGap="1.0" accel="1.0" decel="4.0" />  # 右转车辆
        <route id="N_S" edges="gneE1 -gneE3"/>
        <route id="N_E" edges="gneE1 -gneE0"/>
        <route id="N_W" edges="gneE1 -gneE2"/>
        <route id="S_W" edges="gneE3 -gneE2"/>
        <route id="S_E" edges="gneE3 -gneE0"/>
        <route id="S_N" edges="gneE3 -gneE1"/>
        
        <route id="W_E" edges="gneE2 -gneE0"/>
        <route id="W_N" edges="gneE2 -gneE1"/>
        <route id="W_S" edges="gneE2 -gneE3"/>
        <route id="E_N" edges="gneE0 -gneE1"/>
        <route id="E_S" edges="gneE0 -gneE3"/>
        <route id="E_W" edges="gneE0 -gneE2"/>
        """, file=routes)
        for car_counter, step in enumerate(car_gen_steps):
            straight_or_turn = np.random.uniform()  # 生成一个01随机数、
            if straight_or_turn < 0.5:  # 50%机会直行
                route_straight = np.random.randint(1, 5)
                if route_straight == 1:
                    print(
                        "    <vehicle id='W_E_%i' type='s_car' route='W_E' depart='%s' departLane='random' departSpeed='10' />" % (
                        car_counter, step), file=routes)
                elif route_straight == 2:
                    print(
                        "    <vehicle id='E_W_%i' type='s_car' route='E_W' depart='%s' departLane='random' departSpeed='10' />" % (
                        car_counter, step), file=routes)
                elif route_straight == 3:
                    print(
                        "    <vehicle id='N_S_%i' type='s_car' route='N_S' depart='%s' departLane='random' departSpeed='10' />" % (
                        car_counter, step), file=routes)
                else:
                    print(
                        "    <vehicle id='S_N_%i' type='s_car' route='S_N' depart='%s' departLane='random' departSpeed='10' />" % (
                        car_counter, step), file=routes)
            elif straight_or_turn < 0.75:  # 25%左转
                route_turn = np.random.randint(1, 5)  # 选择一个随机的源目的地
                if route_turn == 1:
                    print(
                        "    <vehicle id='S_W_%i' type='l_car' route='S_W' depart='%s' departLane='random' departSpeed='10' />" % (
                            car_counter, step), file=routes)
                elif route_turn == 2:
                    print(
                        "    <vehicle id='W_N_%i' type='l_car' route='W_N' depart='%s' departLane='random' departSpeed='10' />" % (
                            car_counter, step), file=routes)
                elif route_turn == 3:
                    print(
                        "    <vehicle id='N_E_%i' type='l_car' route='N_E' depart='%s' departLane='random' departSpeed='10' />" % (
                            car_counter, step), file=routes)
                elif route_turn == 4:
                    print(
                        "    <vehicle id='E_S_%i' type='l_car' route='E_S' depart='%s' departLane='random' departSpeed='10' />" % (
                            car_counter, step), file=routes)
            else:  # 25%右转
                route_turn = np.random.randint(1, 5)  # 选择一个随机的源目的地
                if route_turn == 1:
                    print(
                        "    <vehicle id='S_E_%i' type='r_car' route='S_E' depart='%s' departLane='random' departSpeed='10' />" % (
                            car_counter, step), file=routes)
                elif route_turn == 2:
                    print(
                        "    <vehicle id='W_S_%i' type='r_car' route='W_S' depart='%s' departLane='random' departSpeed='10' />" % (
                            car_counter, step), file=routes)
                elif route_turn == 3:
                    print(
                        "    <vehicle id='N_W_%i' type='r_car' route='N_W' depart='%s' departLane='random' departSpeed='10' />" % (
                            car_counter, step), file=routes)
                elif route_turn == 4:
                    print(
                        "    <vehicle id='E_N_%i' type='r_car' route='E_N' depart='%s' departLane='random' departSpeed='10' />" % (
                            car_counter, step), file=routes)
        print("</routes>", file=routes)


l_N = 4000  # 低车流量
l_file_name = 'low_net.rou.xml'
m_N = 8000  # 中车流量
m_file_name = 'mid_net0.rou.xml'
h_N = 10000  # 高车流量
h_file_name = 'high_net0.rou.xml'
MaxStep = 5400
# 不同的转弯比例
lowL = 2
midL = 3
highL = 4
generate_routefile(l_file_name,l_N, MaxStep)
'''
traci.vehicle.getTypeID(vehID)  # 返回给定vehID的车辆类型id
traci.vehicle.setType(vehID,typeID)  #设置给定vehID的车辆typeID
traci.vehicle.setVehicleClass(vehID,class)  #设置给定vehID的vclass

traci.vehicle.getRoute()/getRouteID()/getRouteIndex()
traci.vehicle.setRoute(vehID,edgeList[list]) 将车辆路线改为给定的边列表,列表中的第一条边必须是车辆此刻所在的边。
traci.vehicle.setRouteID(vehID,routeID) 改变给定vehID的路线为routeID

traci.vehicle.moveTo(vehID,laneID,pos[double],reason=0) 含义是车辆vehID移动到一个新的位置 
traci.vehicle.moveToXY(vehID,edgeID,lane,x,y)   将车辆放置在给定的x,y坐标处

?? Q:vclass和vtype分别是什么？
 - vclass是车辆本身的类型abstract vehicle class 默认车辆类为passenger(普通的乘用车)
 - vtype是定义的类型 id区分

traci.lane.setDisallowed(laneID,disallowedClasses)  # 设置该车道不允许通行的车辆vclass
traci.lane.setAllowed(laneID,allowedClasses) #设置该车道允许通行的车辆vclass
'''