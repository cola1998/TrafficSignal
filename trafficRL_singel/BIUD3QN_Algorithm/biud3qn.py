#!/usr/bin/python
# -*- coding: UTF-8 -*-
import traci
import torch
import os
import random
from trafficRL_singel.DQN.recordData import record_data

def main_biud3qn(sumoConfig, biu_buffer, biu_agent, episode, Max_episodes, Max_steps, biu_net_update_times, C, car_N,N):
    biu_R = []
    biu_Waiting_time = []
    biu_delay_time = []
    biu_queue = []

    # 处理sumo
    traci.start(sumoConfig)
    tlsID = traci.trafficlight.getIDList()[0]
    programLogic = traci.trafficlight.getAllProgramLogics(tlsID)
    n = len(programLogic[0].phases)
    new_phase = []
    #
    # tls = (Logic(programID='0', type=0, currentPhaseIndex=0, phases=(
    # Phase(duration=30.0, state='GrrrGGgGGrrrGrrr', minDur=30.0, maxDur=30.0, next=(), name='P0'),
    # Phase(duration=3.0, state='GrrrGyyyGrrrGrrr', minDur=3.0, maxDur=3.0, next=(), name='P01'),
    # Phase(duration=30.0, state='GGgGGrrrGrrrGrrr', minDur=30.0, maxDur=30.0, next=(), name='P1'),
    # Phase(duration=3.0, state='GyyyGrrrGrrrGrrr', minDur=3.0, maxDur=3.0, next=(), name='P11'),
    # Phase(duration=30.0, state='GrrrGrrrGrrrGGgG', minDur=30.0, maxDur=30.0, next=(), name='P2'),
    # Phase(duration=3.0, state='GrrrGrrrGrrrGyyy', minDur=3.0, maxDur=3.0, next=(), name='P21'),
    # Phase(duration=30.0, state='GrrrGrrrGGgGGrrr', minDur=30.0, maxDur=30.0, next=(), name='P3'),
    # Phase(duration=3.0, state='GrrrGrrrGyyyGrrr', minDur=3.0, maxDur=3.0, next=(), name='P31'),
    # Phase(duration=30.0, state='GrrrGrGGGrrrGrGG', minDur=30.0, maxDur=30.0, next=(), name='P4'),
    # Phase(duration=3.0, state='GrrrGryyGrrrGryy', minDur=3.0, maxDur=3.0, next=(), name='P41'),
    # Phase(duration=30.0, state='GrGGGrrrGrGGGrrr', minDur=30.0, maxDur=30.0, next=(), name='P5'),
    # Phase(duration=3.0, state='GryyGrrrGryyGrrr', minDur=3.0, maxDur=3.0, next=(), name='P51')), subParameter={}),)
    # traci.trafficlight.setProgramLogic(tlsID,tls)
    traci.trafficlight.setPhase(tlsID, 0)  # 信号灯相位从0开始
    traci.simulationStep()
    current_state = biu_agent.get_state('queue','turning_number','phase')

    step = 0
    old = biu_net_update_times
    last_action = -1
    count = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        #  and step<Max_steps
        # res= biu_agent.identifyFlow()
        # if res != None:
        #     flowTag = res[0]  # 'k' and 'ra'
        #     flowIndex = int(res[1])  # 0,1,2,3
        # else:  # 表明是平衡车流
        flowTag = 'ba'
        flowIndex = 0

        current_state = torch.tensor(current_state, dtype=torch.float32)
        current_state = current_state.view(1, biu_agent.state_dim)
        action = biu_agent.select_action(current_state, epsilon0=round(episode / Max_episodes, 2))
        print(action)
        if last_action == action:
            count += 1
            if count >= 2:
                while True:
                    x = torch.as_tensor([random.randrange(biu_agent.action_dim)])
                    if x != action:
                        count = 0
                        break
        else:
            count = 0
        last_action = action
        step, reward, waiting_time_t, delay_time1, queue1 = biu_agent.take_action(action, step, flowTag, flowIndex)
        next_state = biu_agent.get_state('queue','turning_number','phase')
        if step >= Max_steps:
            biu_buffer.append_buffer(current_state, action, next_state, reward, 0.0)
        else:
            biu_buffer.append_buffer(current_state, action, next_state, reward, biu_agent.gamma)
        current_state = next_state
        loss = biu_agent.optimize_model()

        biu_R.append(reward)
        biu_Waiting_time.append(waiting_time_t)
        biu_delay_time.append(delay_time1)
        biu_queue.append(queue1)

        biu_net_update_times += 1
        if biu_net_update_times % C == 0:
            biu_agent.optimize_target_model()

    print("biu episode= {0} traci.simulation.getMinExpectedNumber={1}   step={2}".format(episode,
                                                                                         traci.simulation.getMinExpectedNumber(),
                                                                                         step))
    throughoutput = car_N - traci.simulation.getMinExpectedNumber()
    # throughoutput = step
    traci.close()
    #os.mkdir('policy_save/runs/{0}'.format(N))
    os.mkdir('policy_save/runs/{0}/data_record_{1}'.format(N,episode))
    fname = 'policy_save/runs/{0}/data_record_{1}/biu_data_{1}.xlsx'.format(N,episode)
    d = {'reward':biu_R,
         'waiting_time':biu_Waiting_time,
         'delay_time':biu_delay_time,
         'queue':biu_queue}
    record_data(fname,d)
    # 画一个多个子图的
    return round(sum(biu_R) / (biu_net_update_times - old), 2), round(sum(biu_Waiting_time) / car_N, 2), round(
        sum(biu_delay_time) / car_N, 2), round(sum(biu_queue) / car_N, 2), throughoutput, biu_net_update_times
