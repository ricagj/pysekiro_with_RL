from collections import deque
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(tf.config.experimental.get_device_details(gpus[0])['device_name'])

from actions import act
from getvertices import roi

# ---*---

MODEL_WEIGHTS = 'sekiro_weights.h5'
DQN_WEIGHTS = 'dqn_weights.h5'
tmp_WEIGHTS = 'tmp_weights.h5'

# 感兴趣区域大小
ROI_WIDTH = 100
ROI_HEIGHT = 200
FRAME_COUNT = 1

x=190
x_w=290
y=30
y_h=230

# ---*---

def identity_block(input_tensor,out_dim):
    conv1 = tf.keras.layers.Conv2D(out_dim // 4, kernel_size=1, padding="SAME", activation=tf.nn.relu)(input_tensor)
    conv2 = tf.keras.layers.BatchNormalization()(conv1)
    conv3 = tf.keras.layers.Conv2D(out_dim // 4, kernel_size=3, padding="SAME", activation=tf.nn.relu)(conv2)
    conv4 = tf.keras.layers.BatchNormalization()(conv3)
    conv5 = tf.keras.layers.Conv2D(out_dim, kernel_size=1, padding="SAME")(conv4)
    out = tf.keras.layers.Add()([input_tensor, conv5])
    out = tf.nn.relu(out)
    return out
def resnet(width, height, frame_count, output):

    input_xs = tf.keras.Input(shape=[width, height, frame_count])
    
    out_dim = 32
    conv_1 = tf.keras.layers.Conv2D(filters=out_dim,kernel_size=3,padding="SAME",activation=tf.nn.relu)(input_xs)

    out_dim = 16
    identity_1 = tf.keras.layers.Conv2D(filters=out_dim, kernel_size=3, padding="SAME", activation=tf.nn.relu)(conv_1)
    identity_1 = tf.keras.layers.BatchNormalization()(identity_1)
    for _ in range(2):
        identity_1 = identity_block(identity_1,out_dim)

    out_dim = 16
    identity_2 = tf.keras.layers.Conv2D(filters=out_dim, kernel_size=3, padding="SAME", activation=tf.nn.relu)(identity_1)
    identity_2 = tf.keras.layers.BatchNormalization()(identity_2)
    for _ in range(2):
        identity_2 = identity_block(identity_2,out_dim)

    flat = tf.keras.layers.Flatten()(identity_2)
    flat = tf.keras.layers.Dropout(0.5)(flat)
    dense = tf.keras.layers.Dense(32,activation=tf.nn.relu)(flat)
    dense = tf.keras.layers.BatchNormalization()(dense)
    
    logits = tf.keras.layers.Dense(output,activation=tf.nn.softmax)(dense)
    
    model = tf.keras.Model(inputs=input_xs, outputs=logits)
    
    model.compile(
        optimizer = tf.keras.optimizers.Nadam(),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = [
            'accuracy'
        ]
    )
    
    return model

# ---*---

# 奖惩系统
class RewardSystem:
    def __init__(self, capacity, first_status=[152, 0, 99, 0]):

        """
        reward_weights
            生命减少 -     ，reward应为 -，所以对应权重应为 + 。
            架势增加 +     ，reward应为 -，所以对应权重应为 - 。
            敌方生命减少 - ，reward应为 +，所以对应权重应为 - 。
            敌方架势增加 + ，reward应为 +，所以对应权重应为 + 。

            [HP，架势，BossHP，Boss架势] = [1, -1, -1, 1]
        """

        # 把初始状态复制10份填满队列
        self.memory = deque(
            [first_status for _ in range(capacity)],
            maxlen=capacity
        )
        self.capacity = capacity
        
        self.reward_weights = [1, -1, -1, 1]

        # 记录积累reward过程
        self.cur_reward = 0
        self.reward_history = list()

    def store(self, status):

        # 存储新数据，FIFO队列会自动弹出旧数据
        self.memory.append(status)
    
    def get_reward(self):

        # 队列转数组
        data = np.array(self.memory)
        
        # 数组对半分为新数据和旧数据，分别计算均值
        split_n = int(self.capacity / 2)

        past_mean = np.mean(data[:split_n,], axis=0, dtype=np.int16)
        current_mean = np.mean(data[split_n:,], axis=0, dtype=np.int16)
        
        # 差值加权然后将得到的4个 reward 求和
        reward = sum((current_mean - past_mean) * self.reward_weights)
        reward = round(reward, 3)

        self.cur_reward += reward
        self.reward_history.append(self.cur_reward)
        
        return reward

    def plot_reward(self):
        plt.plot(np.arange(len(self.reward_history)), self.reward_history)
        plt.ylabel('reward')
        plt.xlabel('training steps')
        plt.show()

# 经验回放
class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(
            index=range(capacity),
            columns=['screen', 'action', 'reward', 'next_screen']
        )
        self.i = 0
        self.count = 0
        self.capacity = 200000
        
    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
        
    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)

# 智能体
class Sekiro_Agent:
    def __init__(
        self,
        n_action = 5,
        gamma = 0.99,
        replay_memory_size = 200000,
        replay_start_size = 32,
        batch_size = 32,
        update_freq = 5,
        target_network_update_freq = 50

    ):

        self.n_action = n_action
        
        # decay rate 奖励衰减
        self.gamma = gamma

        # 经验回放参数
        self.replay_memory_size = replay_memory_size
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size
        self.replayer = DQNReplayer(replay_memory_size)

        # 奖惩系统
        self.reward_system = RewardSystem(update_freq)

        # 训练评估网络的频率
        self.update_freq = update_freq
        # 更新目标网络的频率
        self.target_network_update_freq = target_network_update_freq

        # 评估网络
        self.evaluate_net = self.build_network()
        # 目标网络
        self.target_net = self.build_network()

        # 初始化计数值
        self.step = 0
        self.fit_count = 0

    def build_network(self):
        model = resnet(ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT,
            output = self.n_action
            )

        if os.path.exists(tmp_WEIGHTS):
            model.load_weights(tmp_WEIGHTS)
        elif os.path.exists(DQN_WEIGHTS):
            model.load_weights(DQN_WEIGHTS)
        else:
            model.load_weights(MODEL_WEIGHTS)

        return model

    # 行为选择
    def choose_action(self, screen):
        screen = roi(screen, x, x_w, y, y_h)
        q_values = self.evaluate_net.predict([screen.reshape(-1, ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT)])[0]
        action = act(q_values)
        return action

    # 学习
    def learn(self, screen, action, reward, next_screen):

        screen = roi(screen, x, x_w, y, y_h)
        next_screen = roi(next_screen, x, x_w, y, y_h)

        # 存储经验
        self.replayer.store(screen, action, reward, next_screen)

        self.step += 1

        if self.step % self.update_freq == 0 and self.replayer.count >= self.replay_start_size:

            # 经验回放
            screens, actions, rewards, next_screens = self.replayer.sample(self.batch_size)

            screens = screens.reshape(-1, ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT)
            next_screens = next_screens.reshape(-1, ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT)

            next_qs = self.target_net.predict(next_screens)
            next_max_qs = next_qs.max(axis=-1)
            targets = self.evaluate_net.predict(screens)
            targets[range(self.batch_size), actions] = rewards + self.gamma * next_max_qs
            self.evaluate_net.fit(screens, targets, verbose=0)
            self.save_evaluate_network()

            if self.step % self.target_network_update_freq == 0:
                self.update_target_network()


    # 更新目标网络权重
    def update_target_network(self):
        self.target_net.load_weights(tmp_WEIGHTS)

    # 保存评估网络权重
    def save_evaluate_network(self):
    	self.evaluate_net.save_weights(tmp_WEIGHTS)