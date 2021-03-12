# https://github.com/ZhiqingXiao/rl-book/blob/master/chapter10_atari/BreakoutDeterministic-v4_tf.ipynb
# https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/DQN3

import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import pandas as pd

from pysekiro.key_tools.actions import act
from pysekiro.img_tools.get_vertices import roi
from pysekiro.model import MODEL

# ---*---

class RewardSystem:
    def __init__(self):
        self.cur_status = None

        self.current_cumulative_reward = 0    # 当前积累的 reward
        self.reward_history = list()    # reward 的积累过程

    # 获取奖励
    def get_reward(self, next_status, cheating_mode=False):
        if sum(next_status) != 0:

            self.next_status = next_status

            # 目的是让Agent尽量维持和积累目标架势。
            # 计算方法：求和[(下一个的状态 - 当前的状态) * 各自的正负强化权重] + 额外奖励
            # 额外奖励：(目标当前架势 -自身当前架势) * 折扣系数
            # 作弊模式不记生命值
            extra_bonus = (self.cur_status[3]- self.cur_status[1]) * 0.01
            if cheating_mode:
                reward = sum((np.array(self.next_status) - np.array(self.cur_status)) * [0, -1,  0, 1]) + extra_bonus
            else:
                reward = sum((np.array(self.next_status) - np.array(self.cur_status)) * [1, -1, -1, 1]) + extra_bonus

            self.cur_status = self.next_status
        else:
            reward = 0
        
        self.current_cumulative_reward += reward
        self.reward_history.append(self.current_cumulative_reward)
        
        return reward

    def save_reward_curve(self, save_path='reward.png'):
        plt.rcParams['figure.figsize'] = 150, 15
        plt.plot(np.arange(len(self.reward_history)), self.reward_history)
        plt.ylabel('reward')
        plt.xlabel('training steps')
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

RESIZE_WIDTH   = 100
RESIZE_HEIGHT  = 100
FRAME_COUNT = 3

# ---*---

class Sekiro_Agent:
    def __init__(
        self,
        n_action, 
        batch_size,
        model_weights = None,
        save_path = None
    ):
        self.n_action = n_action    # 动作数量
        
        self.gamma = 0.99    # 奖励衰减

        self.batch_size = batch_size    # 样本抽取数量
        self.replay_memory_size = 20000    # 记忆容量

        self.epsilon = 1.0                     # 探索参数
        self.epsilon_decrease_rate = 0.9998    # 探索衰减率

        self.update_freq = 50                    # 训练评估网络的频率
        self.target_network_update_freq = 300    # 更新目标网络的频率

        self.model_weights = model_weights    # 指定读取的模型参数的路径
        self.save_path = save_path            # 指定模型权重保存的路径
        if not self.save_path:
            self.save_path = 'tmp_weights.h5'
        
        self.evaluate_net = self.build_network()    # 评估网络
        self.target_net = self.build_network()      # 目标网络
        self.reward_system = RewardSystem()    # 奖惩系统
        self.replayer = DQNReplayer(self.replay_memory_size)    # 经验回放

        self.step = 0    # 计步

    # 评估网络和目标网络的构建方法
    def build_network(self):
        model = MODEL(RESIZE_WIDTH, RESIZE_HEIGHT, FRAME_COUNT,
            outputs = self.n_action,
            model_weights = self.model_weights
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
            action = np.random.randint(self.n_action)
        
        # train = False 直接进入这里
        else:
            screen = cv2.resize(screen, (RESIZE_WIDTH, RESIZE_HEIGHT)).reshape(-1, RESIZE_WIDTH, RESIZE_HEIGHT, FRAME_COUNT)
            q_values = self.evaluate_net.predict(screen)[0]
            action = np.argmax(q_values)
        
        # 执行动作
        act(action)
        
        return action

    # 学习方法
    def learn(self):

        if self.step >= self.batch_size and self.step % self.update_freq == 0:    # 更新评估网络

            if self.step % self.target_network_update_freq == 0:    # 更新目标网络
                print(f'step:{self.step:>4}, current_cumulative_reward:{self.reward_system.current_cumulative_reward:>5.3f}, memory:{self.replayer.count:7>}')
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

    # 更新目标网络权重方法
    def update_target_network(self):
        self.target_net.set_weights(self.evaluate_net.get_weights())

    # 保存评估网络权重方法
    def save_evaluate_network(self):
        self.evaluate_net.save_weights(self.save_path)