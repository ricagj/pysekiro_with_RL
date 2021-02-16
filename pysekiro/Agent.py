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
from pysekiro.getvertices import roi

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


# ---*---

def identity_block(input_tensor,out_dim):
    conv1 = tf.keras.layers.Conv2D(out_dim // 4, kernel_size=1, padding="SAME", activation=tf.nn.relu)(input_tensor)
    conv2 = tf.keras.layers.BatchNormalization()(conv1)
    conv3 = tf.keras.layers.Conv2D(out_dim, kernel_size=1, padding="SAME")(conv2)
    out = tf.keras.layers.Add()([input_tensor, conv3])
    out = tf.nn.relu(out)
    return out
def resnet(width, height, frame_count, output):

    input_xs = tf.keras.Input(shape=[width, height, frame_count])

    out_dim = 8
    conv = tf.keras.layers.Conv2D(filters=out_dim,kernel_size=3,padding="SAME",activation=tf.nn.relu)(input_xs)

    out_dim = 8
    identity = tf.keras.layers.Conv2D(filters=out_dim, kernel_size=3, padding="SAME", activation=tf.nn.relu)(conv)
    identity = tf.keras.layers.BatchNormalization()(identity)
    for _ in range(2):
        identity = identity_block(identity,out_dim)

    flat = tf.keras.layers.Flatten()(identity)
    dense = tf.keras.layers.Dense(16,activation=tf.nn.relu)(flat)
    dense = tf.keras.layers.BatchNormalization()(dense)

    logits = tf.keras.layers.Dense(output,activation=None)(dense)

    model = tf.keras.Model(inputs=input_xs, outputs=logits)

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.MeanSquaredError(),
    )

    return model

# ---*---

initial_state=[152, 0, 99, 0]

class RewardSystem:
    def __init__(self, capacity=4, initial_state=initial_state):
        # 把初始状态复制10份填满队列
        self.memory = deque(
            [initial_state for _ in range(capacity)],
            maxlen=capacity
        )

        # 理论上，容量越大，抗干扰能力越强，但也因此让过去对现在的影响加强
        self.capacity = capacity

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
        self.reward_weights = [0.1, -0.1, -0.1, 0.1] # = [HP，架势，BossHP，Boss架势]

        # 记录积累reward过程
        self.current_cumulative_reward = 0
        self.recording_freq = 0
        self.reward_history = list()

    def store(self, status):

        # 存储新数据，FIFO队列会自动弹出旧数据
        self.memory.append(status)

    def get_reward(self):

        # 队列转numpy数组
        data = np.array(self.memory)

        # 数组对半分为新数据和旧数据，分别计算均值
        split_n = int(self.capacity / 2)

        past_mean = np.mean(data[:split_n,], axis=0, dtype=np.int16)
        current_mean = np.mean(data[split_n:,], axis=0, dtype=np.int16)

        # 差值加权然后将得到的4个 reward 求和
        reward = sum((current_mean - past_mean) * self.reward_weights)

        self.current_cumulative_reward += reward
        self.recording_freq += 1
        if self.recording_freq % 20 == 0:
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
        n_action = 5,      # 动作数量
        batch_size = 8,    # 样本抽取数量
        gamma = 0.99,      # 奖励衰减
        replay_memory_size = 50000,    # 记忆容量
        action_weight = None,    # 动作选择的权重
    ):
        self.n_action = n_action
        self.gamma = gamma
        self.batch_size = batch_size    

        self.evaluate_net = self.build_network()    # 评估网络
        self.target_net = self.build_network()      # 目标网络
        self.reward_system = RewardSystem()                # 奖惩系统
        self.replayer = DQNReplayer(replay_memory_size)    # 经验回放

        if action_weight:
            self.action_weight = action_weight
        else:
            self.action_weight = [1.0 for _ in range(self.n_action)]

    def build_network(self):
        model = resnet(ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT,
            output = self.n_action
        )

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
        screen = roi(screen, x, x_w, y, y_h)
        q_values = self.evaluate_net.predict([screen.reshape(-1, ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT)])[0]
        q_values *= self.action_weight
        action = act(q_values)
        return action

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