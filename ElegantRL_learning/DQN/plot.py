# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import xml.dom.minidom
import numpy as np

def traffic_flow_data_visibliazation(file_name):
    dom = xml.dom.minidom.parse(file_name)
    root = dom.documentElement
    vehicle_list = root.findall('./vehicle')
    for vehicle in vehicle_list:
        pass


'''
 episode
waiting_time
delay_time
reward
queue
'''


def plot_Fixed(max_episodes, queue_list, waiting_time_list):
    x = [i for i in range(max_episodes)]
    plt.title('Fixed Time queue')
    plt.xlabel('episode')
    plt.ylabel('queue')
    plt.plot(x, queue_list)
    plt.savefig('FixedTime_queue_list.jpg')
    plt.show()

    plt.title('Fixed Time waiting time')
    plt.xlabel('episode')
    plt.ylabel('waiting time')
    plt.plot(x, waiting_time_list)
    plt.savefig('FixedTime_waiting_time_list.jpg')
    plt.show()


def plot_BIUD3QN(max_episodes, reward_list, waiting_time_list):
    x = [i for i in range(max_episodes)]
    plt.title('BIU_D3QN reward')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.plot(x, reward_list)
    plt.savefig('BIU_D3QN_reward_list.jpg')
    plt.show()

    plt.title('BIU_D3QN waiting time')
    plt.xlabel('episode')
    plt.ylabel('waiting time')
    plt.plot(x, waiting_time_list)
    plt.savefig('BIU_D3QN_waiting_time_list.jpg')
    plt.show()


def plot_BIUDQN(max_episodes, reward_list, waiting_time_list):
    x = [i for i in range(max_episodes)]
    plt.title('BIU_DQN reward')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.plot(x, reward_list)
    plt.savefig('BIU_DQN_reward_list.jpg')
    plt.show()

    plt.title('BIU_DQN waiting time')
    plt.xlabel('episode')
    plt.ylabel('waiting time')
    plt.plot(x, waiting_time_list)
    plt.savefig('BIU_DQN_waiting_time_list.jpg')
    plt.show()

def plot(title,path,*args,**kwargs):

    # x = np.arange(10)
    # plt.plot(x,x)
    # plt.plot(x,2*x)
    # plt.plot(x,3*x)
    # plt.legend(['y = x', 'y = 2x', 'y = 3x', 'y = 4x'], loc='upper left')
    # plt.show()
    x = np.arange(1,len(kwargs[args[0]])+1)
    legend_list = []
    for tag in args:
        legend_list.append(tag)
        plt.plot(x,kwargs[tag])

    plt.legend(legend_list,loc='upper left')
    plt.title('{0}'.format(title))

    plt.savefig(path+'/'+title)
    plt.show()
# plot('loss',"../../trafficRL_singel/BIUD3QN_Algorithm/policy_save/runs/0_调试",'fixedtime','d3qn','dqn',fixedtime=[5,5,5,5,5],d3qn=[1,2,3,4,5],dqn=[2,4,6,8,10])