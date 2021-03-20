# https://github.com/ZhiqingXiao/rl-book/blob/master/chapter10_atari/BreakoutDeterministic-v4_tf.ipynb
# https://github.com/ZhiqingXiao/rl-book/blob/master/chapter06_approx/MountainCar-v0_tf.ipynb
# https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/DQN3

import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import pandas as pd

from pysekiro.key_tools.actions import act
from pysekiro.model import MODEL

# ---*---

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(
            index=range(capacity),
            columns=['observation', 'action', 'reward', 'next_observation']
        )
        self.i = 0    # index
        self.count = 0    # 代表经验的数量
        self.capacity = capacity    # 经验容量

    def store(self, *args):
        self.memory.loc[self.i] = args    # 存放经验
        self.i = (self.i + 1) % self.capacity    # 更新索引，索引指到尾后，就从头开始。用来替换旧的经验。
        self.count = min(self.count + 1, self.capacity)    # 保证数量不会超过经验容量

    def sample(self, size):    # 这部分我也不是很懂，毕竟这个经验回放的代码不是我写的，总之抽样就对了，好用就对了。
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)

# ---*---

in_depth    = 10
in_height   = 50
in_width    = 50
in_channels = 1
outputs     = 5

# ---*---

# DoubleDQN
class Sekiro_Agent:
    def __init__(
        self,
        lr = 0.01,
        batch_size = 8,
        save_weights_path = None,
        load_weights_path = None

    ):
        self.in_depth    = in_depth       # 时间序列的深度，也可以认为是包含了多少帧图像
        self.in_height   = in_height      # 图像高度
        self.in_width    = in_width       # 图像宽度
        self.in_channels = in_channels    # 颜色通道数量
        self.outputs     = outputs        # 动作数量
        self.lr          = lr,            # 学习率

        self.gamma = 0.99    # 奖励衰减

        self.min_epsilon = 0.3    # 最终探索率

        self.replay_memory_size = 10000    # 记忆容量
        self.replay_start_size = 500       # 开始经验回放时存储的记忆量，到达最终探索率后才开始
        self.batch_size = batch_size       # 样本抽取数量

        self.update_freq = 100                   # 训练评估网络的频率
        self.target_network_update_freq = 500    # 更新目标网络的频率

        self.save_weights_path = save_weights_path    # 指定模型权重参数保存的路径。默认为None，不保存。
        self.load_weights_path = load_weights_path    # 指定模型权重参数加载的路径。默认为None，不加载。

        self.evaluate_net = self.build_network()    # 评估网络
        self.target_net = self.build_network()      # 目标网络
        self.replayer = DQNReplayer(self.replay_memory_size)    # 经验回放

        self.step = 0    # 计步

    # 评估网络和目标网络的构建方法
    def build_network(self):
        model = MODEL(
            in_depth = self.in_depth,
            in_height = self.in_height,
            in_width = self.in_width,
            in_channels = self.in_channels,
            outputs = self.outputs,
            lr = self.lr,
            load_weights_path = self.load_weights_path
        )
        return model

    # 行为选择方法
    def choose_action(self, observation):

        # 先看运行的步数(self.step)有没有达到开始回放经验的要求(self.replay_start_size)，没有就随机探索
                                                  # 如果已经达到了，就再看随机数在不在最终探索率范围内，在的话也是随机探索
        if self.step <= self.replay_start_size or np.random.rand() < self.min_epsilon:
            q_values = np.random.rand(self.outputs)
            self.who_play = '随机探索'
        else:
            observation = observation.reshape(-1, self.in_depth, self.in_height, self.in_width, self.in_channels)
            q_values = self.evaluate_net.predict(observation)[0]
            self.who_play = '模型预测'

        action = np.argmax(q_values)

        # 执行动作
        act(action)

        return action

    # 学习方法
    def learn(self, verbose=0):

        self.step += 1

        # 当前步数满足更新评估网络的要求
        if self.step % self.update_freq == 0:

            # 当前步数满足更新目标网络的要求
            if self.step % self.target_network_update_freq == 0:
                self.update_target_network() 

            # 经验回放
            observations, actions, rewards, next_observations = self.replayer.sample(self.batch_size)

            # 预处理
            observations = observations.reshape(-1, self.in_depth, self.in_height, self.in_width, self.in_channels)
            actions = actions.astype(np.int8)
            next_observations = next_observations.reshape(-1, self.in_depth, self.in_height, self.in_width, self.in_channels)

            # 计算回报的估计值
            # 参考 https://github.com/ZhiqingXiao/rl-book/blob/master/chapter06_approx/MountainCar-v0_tf.ipynb 最后部分 双重深度 Q 网络
            # 与其说参考，不如说Copy得了。。。
            next_eval_qs = self.evaluate_net.predict(next_observations)
            next_actions = next_eval_qs.argmax(axis=-1)

            next_qs = self.target_net.predict(next_observations)
            next_max_qs = next_qs[np.arange(next_qs.shape[0]), next_actions]

            us = rewards + self.gamma * next_max_qs
            targets = self.evaluate_net.predict(observations)
            targets[np.arange(us.shape[0]), actions] = us

            # 学习
            self.evaluate_net.fit(observations, targets, batch_size=1, verbose=verbose)

            # 保存
            self.save_evaluate_network()

    # 更新目标网络权重方法
    def update_target_network(self):
        self.target_net.set_weights(self.evaluate_net.get_weights())

    # 保存评估网络权重方法
    def save_evaluate_network(self):
        self.evaluate_net.save_weights(self.save_weights_path)