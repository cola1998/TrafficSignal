'''
SUMO提供了多种获取统计结果的方式，主要学习两种：
--duration-log.statistics,自动获取实时统计结果   右键show parameter可以看到更多的实时统计信息
    avg.trip length 车辆平均行驶距离
    avg.trip.duration 车辆平均行驶时间
    avg.trip.time loss 平均延迟时间=平均行驶时间-按照期望速度行驶所需时间(16m/s)
    avg.trip.speed 平均车速=平均行驶距离/平均行驶时间

--tripinfo-output ，得到仿真数据文件，在进行后续分析
    仿真结束后，数据都存放在了my_output_file.xml或tripinfo文件中，内容如下
    <tripinfos xmolns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/tripinfo_file.xsd">
        ...
        <tripinfo id="N_E_2" depart="33.00" departLane="gneE1_3" departPos="3.10" departSpeed="10.00" departDelay="0.00" arrival="48.40" arrivalLane="gneE0_3" arrivalPos="83.20" arrivalSpeed="14.02" duration="15.40" routeLength="192.98" waitingTime="0.00" waitingCount="0" stopTime="0.00" timeLoss="1.15" rerouteNo="0" devices="tripinfo_N_E_2" vType="standard_car" speedFactor="1.01" vaporized=""/>
        <tripinfo id="W_E_0" depart="29.00" departLane="gneE2_2" departPos="3.10" departSpeed="10.00" departDelay="0.00" arrival="76.30" arrivalLane="gneE0_0" arrivalPos="83.20" arrivalSpeed="13.25" duration="47.30" routeLength="197.00" waitingTime="21.90" waitingCount="1" stopTime="0.00" timeLoss="33.91" rerouteNo="0" devices="tripinfo_W_E_0" vType="standard_car" speedFactor="1.06" vaporized=""/>
        <tripinfo id="S_E_5" depart="60.00" departLane="gneE3_2" departPos="3.10" departSpeed="10.00" departDelay="0.00" arrival="79.20" arrivalLane="gneE0_0" arrivalPos="83.20" arrivalSpeed="11.51" duration="19.20" routeLength="172.33" waitingTime="0.00" waitingCount="0" stopTime="0.00" timeLoss="3.33" rerouteNo="0" devices="tripinfo_S_E_5" vType="standard_car" speedFactor="0.83" vaporized=""/>
        <tripinfo id="W_S_8" depart="68.00" departLane="gneE2_1" departPos="3.10" departSpeed="10.00" departDelay="0.00" arrival="85.50" arrivalLane="-gneE3_0" arrivalPos="83.20" arrivalSpeed="12.58" duration="17.50" routeLength="172.33" waitingTime="0.00" waitingCount="0" stopTime="0.00" timeLoss="3.40" rerouteNo="0" devices="tripinfo_W_S_8" vType="standard_car" speedFactor="0.94" vaporized=""/>
        <tripinfo id="N_W_10" depart="75.00" departLane="gneE1_3" departPos="3.10" departSpeed="10.00" departDelay="0.00" arrival="92.30" arrivalLane="-gneE2_0" arrivalPos="83.20" arrivalSpeed="13.13" duration="17.30" routeLength="172.33" waitingTime="0.00" waitingCount="0" stopTime="0.00" timeLoss="5.08" rerouteNo="0" devices="tripinfo_N_W_10" vType="standard_car" speedFactor="1.09" vaporized=""/>
        ...
    </tripinfos>

    可以借助xml.etree.ElementTree module进行分析  https://docs.python.org/3/library/xml.etree.elementtree.html#module-xml.etree.ElementTree
'''

'''
SUMO中的车辆动力学模型 -- 两种：纵向和横向
1. longitudinal model：纵向动力学模型，描述车辆加速或者减速
    近似看作质点，采用比较简单的car-following model（跟车模型）来描述车辆速度和位置变化规律。
    car-following model中包含两种情况：无前车和有前车。
    1）对于无前车的情形，车辆保持最大速度，这里的最大速度考虑三方面的因素：
        (1) 该类型车辆本身能达到的最大物理速度
        (2) 前一时刻速度经过最大加速后在当前时刻所能达到的最大速度
        (3) 当前行驶道路规定的最大速度
    最终的最大速度定为这三者中的最小值。
    2）对于有前车的清形，则要计算安全的行驶速度，保证任何情况下（尤其是前车紧急刹车时）车辆不会相撞。
        用不同的car-following model的主要区别就是如何计算安全的行驶速度。目前sumo中采用的是改进的Krauss model
2. lateral model：横向动力学模型，描述车辆换道
sumo采用的是lane changing model，简单的说就是以决策树的方式设定诸多换道条件，只要满足某些条件，就进行相应的换道操作。
默认的lane changing model是瞬间换道，即在一个simulation step中完成换道，直观的看就是车辆在两个车道之间瞬移。
更加精细的模型包括：Sublane Model、Simple Continous lane-change model


'''