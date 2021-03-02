from collections import deque
import os

import matplotlib.pyplot as plt
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import pandas as pd

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(tf.config.experimental.get_device_details(gpus[0])['device_name'])

from pysekiro.actions import act
from pysekiro.get_vertices import roi
from pysekiro.model import resnet

# ---*---

MODEL_WEIGHTS = 'sekiro_weights.h5'
DQN_WEIGHTS = 'dqn_weights.h5'
tmp_WEIGHTS = 'tmp_weights.h5'

ROI_WIDTH = 100
ROI_HEIGHT = 200
FRAME_COUNT = 1

x=190
x_w=290
y=30
y_h=230

n_action = 5

# ---*---

# 根据 actions.py
action_point = {
    0:  0.15,    # 攻击
    1:  0.12,    # 弹反
    2:  0.1,    # 垫步
    3:  0.1,    # 跳跃
    4: -0.05    # 其他
}

class RewardSystem:
    def __init__(self):
        """     
        设置 正强化 和 负强化
        由于计算 reward 的算法是 现在的状态减去过去的状态，所以
            类型     | 状态 | reward | 权重正负 |
            我方生命 |  +   |   +    |    +    |
            我方生命 |  -   |   -    |    +    |
            我方架势 |  +   |   -    |    -    |
            我方架势 |  -   |   +    |    -    |
            敌方生命 |  +   |   -    |    -    |
            敌方生命 |  -   |   +    |    -    |
            敌方架势 |  +   |   +    |    +    |
            敌方架势 |  -   |   -    |    +    |
        """
        self.reward_weights = [0.1, -0.1, -0.1, 0.1] # = [自身HP，自身架势，目标HP，目标架势]

        self.past_status = [153, 0, 101, 0]

        # 记录积累reward过程
        self.current_cumulative_reward = 0
        self.reward_history = list()

    def get_reward(self, status, action):

        self.status = status
        point = action_point[action]
        reward = sum((np.array(self.status) - np.array(self.past_status)) * self.reward_weights)
        self.past_status = self.status
        self.current_cumulative_reward += reward + point
        self.reward_history.append(self.current_cumulative_reward)

        return reward

    def save_reward_curve(self, save_path='reward.png'):
        plt.plot(np.arange(len(self.reward_history)), self.reward_history)
        plt.ylabel('reward')
        plt.xlabel('training steps')
        plt.savefig(save_path)

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

class Sekiro_Agent:
    def __init__(
        self,
        n_action = n_action,      # 动作数量
        batch_size = 8,    # 样本抽取数量
        gamma = 0.99,      # 奖励衰减
        epsilon = 1.0,
        epsilon_decrease_rate = 0.9995,
        replay_memory_size = 50000,    # 记忆容量
        load_weights = False
    ):
        self.n_action = n_action
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decrease_rate = epsilon_decrease_rate
        self.batch_size = batch_size
        self.load_weights = load_weights

        self.evaluate_net = self.build_network()    # 评估网络
        self.target_net = self.build_network()      # 目标网络
        self.reward_system = RewardSystem()                # 奖惩系统
        self.replayer = DQNReplayer(replay_memory_size)    # 经验回放

    def build_network(self):
        model = resnet(ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT,
            outputs = self.n_action
        )
        if self.load_weights:
            if os.path.exists(DQN_WEIGHTS):
                model.load_weights(DQN_WEIGHTS)
                print('Load ' + DQN_WEIGHTS)
            elif os.path.exists(MODEL_WEIGHTS):
                model.load_weights(MODEL_WEIGHTS)
                print('Load ' + MODEL_WEIGHTS)
            else:
                print('Nothing to load')

        return model

    # 行为选择
    def choose_action(self, screen):
        if np.random.rand() < self.epsilon:
            q_values = np.random.randint(self.n_action)
            self.epsilon *= self.epsilon_decrease_rate
        else:
            screen = roi(screen, x, x_w, y, y_h)
            q_values = self.evaluate_net.predict([screen.reshape(-1, ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT)])[0]
            q_values = np.argmax(q_values)

        act(q_values)
        return q_values

    # 学习
    def learn(self):

        # 经验回放
        screens, actions, rewards, next_screens = self.replayer.sample(self.batch_size)

        screens = screens.reshape(-1, ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT)
        next_screens = next_screens.reshape(-1, ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT)

        next_qs = self.target_net.predict(next_screens)
        next_max_qs = next_qs.max(axis=-1)
        targets = self.evaluate_net.predict(screens)
        targets[range(self.batch_size), actions] = rewards + self.gamma * next_max_qs

        self.evaluate_net.fit(screens, targets, verbose=0)

    # 更新目标网络权重
    def update_target_network(self, load_path=tmp_WEIGHTS):
        self.target_net.load_weights(load_path)

    # 保存评估网络权重
    def save_evaluate_network(self, save_path=tmp_WEIGHTS):
        try:
            self.evaluate_net.save_weights(save_path)
        except:
            print('save weights faild!!!')