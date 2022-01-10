#!/usr/bin/python
# -*- coding: UTF-8 -*-
import traci
import torch
# 单独一轮的
def main_dqn(sumoConfig,dqn_buffer,dqn_agent,episode,Max_episodes,Max_steps,dqn_net_update_times,C,car_N):
    dqn_R = 0
    dqn_Waiting_time = 0
    dqn_delay_time = 0
    dqn_queue = 0

    # 处理sumo
    traci.start(sumoConfig)
    tlsID = traci.trafficlight.getIDList()[0]
    traci.trafficlight.setPhase(tlsID, 0)  # 信号灯相位从0开始
    traci.simulationStep()
    current_state = dqn_agent.get_state('queue','phase')

    step = 0
    old = dqn_net_update_times
    while traci.simulation.getMinExpectedNumber() > 0 :

        current_state = torch.tensor(current_state, dtype=torch.float32)
        current_state = current_state.view(1, dqn_agent.state_dim)
        action = dqn_agent.select_action(current_state)
        step, reward, waiting_time_t, delay_time1, queue1 = dqn_agent.take_action(action, step)
        next_state = dqn_agent.get_state('queue','phase')

        if step >= Max_steps:
            dqn_buffer.append_buffer(current_state, action, next_state, reward, 0.0)
        else:
            dqn_buffer.append_buffer(current_state, action, next_state, reward, dqn_agent.gamma)
        current_state = next_state
        loss = dqn_agent.optimize_model()

        dqn_R += reward
        dqn_Waiting_time += waiting_time_t
        dqn_delay_time += delay_time1
        dqn_queue += queue1

        dqn_net_update_times += 1
        if dqn_net_update_times % C == 0:
            dqn_agent.optimize_target_model()

    print("dqn episode= {0} 结束时traci.simulation.getMinExpectedNumber={1} 和 step={2}".format(episode,
                                                                                        traci.simulation.getMinExpectedNumber(),
                                                                                        step))

    throughOutput = car_N - traci.simulation.getMinExpectedNumber()
    # throughOutput = step
    traci.close()
    return round(dqn_R/(dqn_net_update_times-old), 2),round(dqn_Waiting_time / car_N, 2),round(dqn_delay_time / car_N, 2),round(dqn_queue / car_N, 2),throughOutput,dqn_net_update_times
