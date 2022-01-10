'''
1、初始化
2、while循环控制训练过程
'''
args = Arguments()   # 加载默认的超参数
env = PreprocessEnv()  # 创建gym环境
agent = agent.XX  # 基于算法创建Agent
evalutor = Evaluator()   # 用于评测并保存模型
buffer = ReplayBuffer()  # 回放缓存

while True:
    agent.explore_env()  #agent 探索环境 产生transitions并存放到replay buffer
    agent.update_net()  # agent根据回放缓存 replay buffer 中的一批数据来更新网络参数
    evaluator.evaluate_save()   # 测评agent的表现，保存具有最高得分的模型参数