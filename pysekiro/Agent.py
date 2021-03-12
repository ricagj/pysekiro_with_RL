# https://github.com/ZhiqingXiao/rl-book/blob/master/chapter10_atari/BreakoutDeterministic-v4_tf.ipynb
# https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/DQN3

import matplotlib.pyplot as plt
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import pandas as pd

from pysekiro.key_tools.actions import act
from pysekiro.model import MODEL

# ---*---

class RewardSystem:
    def __init__(self):
        self.cur_status = None

        self.total_reward = 0    # 当前积累的 reward
        self.reward_history = list()    # reward 的积累过程

    # 获取奖励
    def get_reward(self, next_status):
        if sum(next_status) != 0:
            # 计算方法：求和[(下一个的状态 - 当前的状态) * 各自的正负强化权重]
            self.next_status = next_status
            reward = sum((np.array(self.next_status) - np.array(self.cur_status)) * [0.01, -0.01, -0.01, 0.01])
            self.cur_status = next_status
        else:
        	self.cur_status = next_status
            reward = 0

        self.total_reward += reward
        self.reward_history.append(self.total_reward)

        return reward

    def save_reward_curve(self, save_path='reward.png'):
        plt.rcParams['figure.figsize'] = 100, 15
        total = len(self.reward_history)
        plt.plot(np.arange(total), self.reward_history)
        plt.ylabel('reward')
        plt.xlabel('training steps')
        plt.xticks(np.arange(0, total, int(total/100)))
        plt.savefig(save_path)
        plt.show()

# ---*---

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(
            index=range(capacity),
            columns=['screen', 'action', 'reward', 'next_screen']
        )
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)

# ---*---

RESIZE_WIDTH   = 50
RESIZE_HEIGHT  = 50
FRAME_COUNT    = 3

# ---*---

class Sekiro_Agent:
    def __init__(
        self,
        load_weights_path = None,
        save_weights_path = None
    ):
        self.n_action = 5    # 动作数量
        
        self.gamma = 0.99    # 奖励衰减

        self.replay_memory_size = 200000    # 记忆容量
        self.replay_start_size = 5000       # 开始经验回放时存储的记忆量
        self.batch_size = 16                # 样本抽取数量

        self.epsilon = 1.0                    # 探索参数
        self.epsilon_decrease_rate = 0.99954  # 探索衰减率

        self.update_freq = 100                   # 训练评估网络的频率
        self.target_network_update_freq = 400    # 更新目标网络的频率

        self.load_weights_path = load_weights_path    # 指定读取的模型参数的路径
        self.save_weights_path = save_weights_path    # 指定模型权重保存的路径

        self.evaluate_net = self.build_network()    # 评估网络
        self.target_net = self.build_network()      # 目标网络
        self.reward_system = RewardSystem()    # 奖惩系统
        self.replayer = DQNReplayer(self.replay_memory_size)    # 经验回放

        self.step = 1    # 计步

    # 评估网络和目标网络的构建方法
    def build_network(self):
        model = MODEL(
        	width = RESIZE_WIDTH,
        	height = RESIZE_HEIGHT,
        	frame_count = FRAME_COUNT,
            outputs = self.n_action,
            load_weights_path = self.load_weights_path
        )
        return model

    # 行为选择与执行方法
    def choose_action(self, screen, train):
        if train:
            r = np.random.rand()
        else:
            r = 1.01    # 永远大于 self.epsilon

        # train = True 开启探索模式
        if r < self.epsilon:
            self.epsilon *= self.epsilon_decrease_rate    # 逐渐减小探索参数, 降低行为的随机性
            q_values = np.random.rand(self.n_action) * [0.3, 0.25, 0.2, 0.1, 0.15]

        # train = False 直接进入这里
        else:
            screen = screen.reshape(-1, RESIZE_WIDTH, RESIZE_HEIGHT, FRAME_COUNT)
            q_values = self.evaluate_net.predict(screen)[0]

        action = np.argmax(q_values)

        # 执行动作
        act(action)

        return action

    # 学习方法
    def learn(self):

        if self.replayer.count >= self.replay_start_size and self.step % self.update_freq == 0:    # 更新评估网络

            if self.step % self.target_network_update_freq == 0:    # 更新目标网络
                print(f'\rstep:{self.step:>4}, total_reward:{self.reward_system.total_reward:>5.3f}, memory:{self.replayer.count:7>}')
                self.update_target_network() 

            # 经验回放
            screens, actions, rewards, next_screens = self.replayer.sample(self.batch_size)

            screens = screens.reshape(-1, RESIZE_WIDTH, RESIZE_HEIGHT, FRAME_COUNT)
            next_screens = next_screens.reshape(-1, RESIZE_WIDTH, RESIZE_HEIGHT, FRAME_COUNT)

            # 计算回报的估计值
            q_next = self.target_net.predict(next_screens)
            q_target = self.evaluate_net.predict(screens)
            q_target[range(self.batch_size), actions] = rewards + self.gamma * q_next.max(axis=-1)

            self.evaluate_net.fit(screens, q_target, verbose=0)

            self.save_evaluate_network()

        self.step += 1

    # 更新目标网络权重方法
    def update_target_network(self):
        self.target_net.set_weights(self.evaluate_net.get_weights())

    # 保存评估网络权重方法
    def save_evaluate_network(self):
        self.evaluate_net.save_weights(self.save_weights_path)