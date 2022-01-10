import random
from collections import namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward','mask'))  # 命名元组
class ReplayBuffer(object):
    def __init__(self, capacity, minibatch):
        '''
        M 容量
        m 最少条数
        minibatch  采样大小
        需不需要设置优先级啥的
        '''
        self.memory = []
        self.capacity = capacity
        self.minibatch = minibatch
        self.position = 0

    def append_buffer(self,*args):
        # record = [st,at,rt,st+1]
        if len(self.memory) >= self.capacity:
            # remove 某条记录
            self.memory.pop(0)  # 删除最旧的一条记录
        else:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        if None in self.memory:
            self.memory.remove(None)
    def get_size(self):
        return len(self.memory)

    def sample(self):
        trajs = random.sample(self.memory,self.minibatch)
        return trajs
