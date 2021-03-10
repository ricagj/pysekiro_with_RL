import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import pandas as pd

from pysekiro.actions import act
from pysekiro.get_status import get_status
from pysekiro.get_vertices import roi
from pysekiro.model import MODEL

# ---*---

ROI_WIDTH   = 100
ROI_HEIGHT  = 100
FRAME_COUNT = 3

x   = 140
x_w = 340
y   = 30
y_h = 230

n_action = 5

# ---*---

# 约束状态值的上下限，防止异常值和特殊值的影响。
def limit(value, lm1, lm2):

    if value < lm1:
        return lm1
    elif value > lm2:
        return lm2
    else:
        return value

# ---*---

class RewardSystem:
    def __init__(self):
        self.cur_status = [152, 0, 100, 0]

        self.current_cumulative_reward = 0    # 当前积累的 reward
        self.reward_history = list()    # reward 的积累过程

    def get_reward(self, next_status):
        if sum(status) != 0:

            self.next_status = next_status

            # 自身状态的计算方法：(下一个的状态 - 当前的状态) * 正负强化权重
            s1 = (self.next_status[0] - self.cur_status[0]) *  1    # 自身生命
            s2 = (self.next_status[1] - self.cur_status[1]) * -1    # 自身架势

            # 目标状态的计算方法：约束上下限(下一个的状态 - 当前的状态) * 正负强化权重
            t1 = limit((self.next_status[2] - self.cur_status[2]), -100,   0) * -1    # 目标生命
            t2 = limit((self.next_status[3] - self.cur_status[3]),  -20, +20) *  1    # 目标架势

            # 分开生命值和架势并赋予这样的权重是为了降低生命值状态变化的影响
            reward = 0.1 * (s1 + t1) + 0.9 * (s2 + t2)
            # print(f'\t s1:{s1:>4}, s2:{s2:>4}, t1:{t1:>4}, t2:{t2:>4}, reward:{reward:>4}')

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

def get_data_quality():
    
    reward_system = RewardSystem()
    
    path1 = 'The_battle_memory'
    path2 = 'Data_quality'
    
    for target in os.listdir(path1):
        save_dir = os.path.join(path2, target)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        data_list = os.listdir(os.path.join(path1, target))

        m = max([int(re.findall('\d+', x)[0]) for x in data_list])
        for i in range(1, m+1):
            filename = f'training_data-{i}.npy'
            data_path = os.path.join(path1, target, filename)

            if os.path.exists(data_path):
                dataset = np.load(data_path, allow_pickle=True)

                reward_system.cur_status = get_status(dataset[0][0])
                for step in range(1, len(dataset)):
                    reward_system.get_reward(get_status(dataset[step][0]))
                reward_system.save_reward_curve(save_path=os.path.join(path2, target, filename[:-4]+'.png'))
                
                reward_system.current_cumulative_reward = 0
                reward_system.reward_history = list()
                print(data_path, 'done')
            else:
                print(f'{filename} does not exist ')

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
        batch_size = 128,
        replay_memory_size = 20000,
        epsilon = 1.0,
        epsilon_decrease_rate = 0.999,
        update_freq = 100,
        target_network_update_freq = 300,
        model_weights = None,
        save_path = None
    ):
        self.n_action = n_action    # 动作数量
        
        self.gamma = gamma    # 奖励衰减

        self.batch_size = batch_size                    # 样本抽取数量
        self.replay_memory_size = replay_memory_size    # 记忆容量

        self.epsilon = epsilon                                # 探索参数
        self.epsilon_decrease_rate = epsilon_decrease_rate    # 探索衰减率

        self.update_freq = update_freq    # 训练评估网络的频率
        self.target_network_update_freq = target_network_update_freq    # 更新目标网络的频率

        self.model_weights = model_weights    # 指定读取的模型参数的路径
        self.save_path = save_path            # 指定模型权重保存的路径
        if not self.save_path:
            self.save_path = 'tmp_weights.h5'
        
        self.evaluate_net = self.build_network()    # 评估网络
        self.target_net = self.build_network()      # 目标网络
        self.reward_system = RewardSystem()                     # 奖惩系统
        self.replayer = DQNReplayer(self.replay_memory_size)    # 经验回放

        self.step = 0    # 计步

    # 评估网络和目标网络的构建方法
    def build_network(self):
        model = MODEL(ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT,
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
            screen = cv2.resize(roi(screen, x, x_w, y, y_h), (ROI_WIDTH, ROI_HEIGHT)).reshape(-1, ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT)
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

            screens = screens.reshape(-1, ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT)
            next_screens = next_screens.reshape(-1, ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT)

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