import numpy as np
import tensorflow as tf
import gym

np.random.seed(2)
tf.set_random_seed(2)

# 超参数
OUTPUT_GRAPH = False
MAX_EPISODE = 3000  # max episode
DISPLAY_REWARD_THRESHOLD = 200  # 刷新阈值
MAX_EP_STEP = 10000  # max step
RENDER = False  # 渲染开关
GAMMA = 0.9  # 衰变值
LR_A = 0.001  # actor 学习率
LR_C = 0.01  # critic 学习率

# 环境
env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

N_F = env.observation_space.shape[0]  # 状态空间
N_A = env.action_space.n  # 动作空间


class Actor(object):
    def __init__(self, sess, n_feature, n_actions, lr=0.001):
        # 使用 tensorflow 建立Actor神经网络
        # 搭建好训练的graph
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_feature], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")

        with tf.variable_scope('Actor'):  # 用于定义创建变量(层)的操作的上下文管理器.
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )
            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])  # tf.log(y)  计算元素 y 的自然对数, y=ex
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage td_error guided loss  计算张量的各个维度上的元素的平均值.

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v)=maximize(exp_v)
            # Adam优化算法: 是一个寻找全局最优点的优化算法, 引入了二次方梯度校正.

    def learn(self, s, a, td):
        # s,a 用于产生Gradient ascent的方向
        # td来自critic，用于告诉actor这方向不对
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        # 在执行sess.run()时，tensorflow并不是计算了整个图，只是计算了与想要fetch的值相关的部分。
        return exp_v

    def choose_action(self, s):
        # 根据s选择行为a
        s = s[np.newaxis, :]  # 在np.newaxis这一位置增加一个1维？(5,) -> (1,5)
        probs = self.sess.run(self.acts_prob, {self.s: s})  # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())  # return a choice


class Critic(object):
    def __init__(self, sess, n_feature, lr=0.01):
        # 用tensorflow建立critic网络
        # 搭建好训练的graph
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_feature], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,
                activation=tf.nn.relu,   # None
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )
            self.v = tf.layers.dense(
                inputs=l1,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='V'
            )
        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)  # TD_error = (r*gamma*V_next) - V_eval

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        # 学习状态的价值！(state,value)
        # 计算TD_error = (r+v_)-v
        # 用TD_error评判这一步的行为有没有带来比平时更好的结果
        # 可以把它看作是advantage
        # return 学习产生的TD_error
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.s: s, self.v_: v_, self.r: r})
        return td_error


sess = tf.Session()
actor = Actor(sess, n_feature=N_F, n_actions=N_A, lr=LR_A)  # 初始化actor
critic = Critic(sess, n_feature=N_F, lr=LR_C)  # 初始化critic  LR_C=0.1 LR_A=0.01 我们需要一个好的老师，因此老师的学习率需要高一点
sess.run(tf.global_variables_initializer())  # 初始化参数

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)  # 输出日志

# 开始迭代过程 对应伪代码部分
for i_episode in range(MAX_EPISODE):
    s = env.reset()  # 环境初始化
    t = 0
    track_r = []  # 记录每个回合的奖励
    while True:
        if RENDER:
            env.render()
        a = actor.choose_action(s)  # actor选取动作
        s_, r, done, info = env.step(a)  # 环境反馈

        if done:
            r = -20  # 回合结束的惩罚？？

        track_r.append(r)  # 记录奖励

        td_error = critic.learn(s, r, s_)  # critic 学习
        actor.learn(s, a, td_error)  # actor学习
        s = s_
        t += 1

        if done or t >= MAX_EP_STEP:
            # 回合结束 打印回合奖励
            ep_rs_sum = sum(track_r)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True  # Rendering????
            print("episode: ", i_episode, " reward:", int(running_reward))
            break

'''
每轮过程
for episode in max_episode:
    初始化环境信息
    θ,w = 0
    t = 0
    while t<max_step: 
        actor选择动作
        执行并获得环境反馈
        
        critic网络更新
        actor网络更新
        s = s_
        t = t + 1
'''
