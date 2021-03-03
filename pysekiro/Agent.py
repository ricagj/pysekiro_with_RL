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
TMP_WEIGHTS = 'tmp_weights.h5'

ROI_WIDTH = 100
ROI_HEIGHT = 200
FRAME_COUNT = 1

x=190
x_w=290
y=30
y_h=230

n_action = 5

# ---*---

class RewardSystem:
    def __init__(self):
        self.reward_weights = [0.1, -0.1, -0.1, 0.1] # = [自身生命，自身架势，目标生命，目标架势]
        self.past_status = [152, 0, 100, 0]
        # self.action_point = {
        #     0: 1,    # 攻击
        #     1: 1,    # 弹反
        #     2: 1,    # 垫步
        #     3: 1,    # 跳跃
        #     4: 1    # 其他
        # }

        # 记录积累reward过程
        self.current_cumulative_reward = 0
        self.reward_history = list()

    def get_reward(self, status, action):

        # 计算 现在的状态 - 过去的状态 的差值，然后把现在的状态赋值给self.past_status
        self.status = status
        status_difference = np.array(self.status) - np.array(self.past_status)
        self.past_status = self.status

        # 差值乘上正负强化的权重并求和
        reward = sum(status_difference * self.reward_weights)    ## * self.action_point[action]

        self.current_cumulative_reward += reward
        self.reward_history.append(self.current_cumulative_reward)

        return reward

    def save_reward_curve(self, save_path='reward.png'):
        plt.rcParams['figure.figsize'] = 30, 15
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
        n_action = n_action, 
        gamma = 0.99,
        batch_size = 8,
        replay_memory_size = 50000,
        epsilon = 1.0,
        epsilon_decrease_rate = 0.999,
        model_weights = None
    ):
        self.n_action = n_action    # 动作数量
        
        self.gamma = gamma    # 奖励衰减

        self.batch_size = batch_size                    # 样本抽取数量
        self.replay_memory_size = replay_memory_size    # 记忆容量

        self.epsilon = epsilon                                # 探索参数
        self.epsilon_decrease_rate = epsilon_decrease_rate    # 探索衰减率

        self.model_weights = model_weights    # 指定读取的模型参数的路径

        self.evaluate_net = self.build_network()    # 评估网络
        self.target_net = self.build_network()      # 目标网络
        self.reward_system = RewardSystem()                # 奖惩系统
        self.replayer = DQNReplayer(self.replay_memory_size)    # 经验回放

    # 构建网络
    def build_network(self):
        model = resnet(ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT,
            outputs = self.n_action
        )
        if self.model_weights:
            if os.path.exists(self.model_weights):
                model.load_weights(self.model_weights)
                print('Load ' + self.model_weights)
            else:
                print('Nothing to load')

        return model

    # 行为选择
    def choose_action(self, screen, train=False):
        r = 1
        if train:
            r = np.random.rand()
        
        if r < self.epsilon:
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
    def update_target_network(self, load_path=TMP_WEIGHTS):
        self.target_net.load_weights(load_path)

    # 保存评估网络权重
    def save_evaluate_network(self, save_path=TMP_WEIGHTS):
        try:
            self.evaluate_net.save_weights(save_path)
        except:
            print('save weights faild!!!')