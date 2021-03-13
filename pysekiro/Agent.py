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
            reward = sum((np.array(self.next_status) - np.array(self.cur_status)) * [0.1, -0.1, -0.1, 0.1])
            
            # Boss 把我们架势打没了还加分就离谱
            Self_Posture_status = self.next_status[1] - self.cur_status[1]
            if abs(Self_Posture_status) > 100:    # 特殊情况：当前架势变化超过100时很可能是被Boss击败了，所以不计自身架势这部分奖励，还有复活和击败boss时的也不计
                reward -= Self_Posture_status * -0.1

        else:
            reward = 0

        self.cur_status = next_status

        self.total_reward += reward
        self.reward_history.append(self.total_reward)

        return reward

    def save_reward_curve(self, save_path='reward.png'):
        total = len(self.reward_history)
        if total > 100:
            plt.rcParams['figure.figsize'] = 100, 15
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

        self.replay_memory_size = 75000    # 记忆容量
        self.replay_start_size  = 5000      # 开始经验回放时存储的记忆量
        self.batch_size = 64                # 样本抽取数量

        self.epsilon = 1.0                    # 初始探索率
        self.epsilon_decrease_rate = 0.9999   # 探索衰减率
        self.min_epsilon = 0.1                # 最终探索率

        self.update_freq = 100                   # 训练评估网络的频率，约20秒
        self.target_network_update_freq = 300    # 更新目标网络的频率，约60秒

        self.load_weights_path = load_weights_path    # 指定模型权重参数加载的路径。默认为None，不加载。
        self.save_weights_path = save_weights_path    # 指定模型权重参数保存的路径。默认为None，不保存。注：默认也是测试模式，若设置该参数，就会开启训练模式

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
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decrease_rate    # 逐渐减小探索参数, 降低行为的随机性
            else:
                self.epsilon = self.min_epsilon
            q_values = np.random.rand(self.n_action)

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

        # 条件之一：记忆量符合开始经验回放时需要存储的记忆量
        if self.replayer.count >= self.replay_start_size and self.step % self.update_freq == 0:    # 更新评估网络
            
            if self.step % self.target_network_update_freq == 0:    # 更新目标网络
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