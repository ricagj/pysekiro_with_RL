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

            self.next_status = next_status

            # 计算方法：求和[(下一个的状态 - 当前的状态) * 各自的正负强化权重]

            s1 = min(0, self.next_status[0] - self.cur_status[0]) *  1    # 自身生命，不计增加
            t1 = min(0, self.next_status[2] - self.cur_status[2]) * -1    # 目标生命，不计增加
            
            s2 = self.next_status[1] - self.cur_status[1]    # 自身架势
            s2 = s2 * -1 if abs(s2) < 100 else 0

            t2 = self.next_status[3] - self.cur_status[3]    # 目标架势
            t2 = t2 *  1 if abs(s2) < 200 else 0

            reward = s1 + s2 + t1 + t2

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
            columns=['observation', 'action', 'reward', 'next_observation']
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
        width
        height,
        frame_count,
        n_action,
        learning_rate,
        load_weights_path = None,
        save_weights_path = None
    ):
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self.outputs = n_action    # 动作数量
        self.lr = learning_rate    # 学习率

        self.gamma = 0.90    # 奖励衰减

        self.epsilon = 1.0           # 初始探索率
        self.min_epsilon = 0.4       # 最终探索率
        self.epsilon_step = 3000    # 到达最终探索率前的步数
        self.epsilon_decrease_rate = (self.epsilon - self.min_epsilon) * self.epsilon_step    # 探索衰减率

        self.replay_memory_size = 20000               # 记忆容量
        self.replay_start_size = self.epsilon_step    # 开始经验回放时存储的记忆量，到达最终探索率后才开始
        self.batch_size = 256                         # 样本抽取数量

        self.update_freq = 200                   # 训练评估网络的频率
        self.target_network_update_freq = 1000    # 更新目标网络的频率

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
            width = self.width,
            height = self.height,
            frame_count = self.frame_count,
            outputs = self.outputs,
            lr = self.lr,
            load_weights_path = self.load_weights_path
        )
        return model

    # 行为选择与执行方法
    def choose_action(self, observation, train):
        if train:
            r = np.random.rand()
        else:
            r = 1.01    # 永远大于 self.epsilon

        # train = True 开启探索模式
        if r < self.epsilon:
            if self.epsilon > self.min_epsilon:
                self.epsilon -= self.epsilon_decrease_rate    # 逐渐减小探索参数, 降低行为的随机性

            q_values = np.random.rand(self.n_action)

        # train = False 直接进入这里
        else:
            observation = observation.reshape(-1, self.width, self.height, self.frame_count)
            q_values = self.evaluate_net.predict(observation)[0]

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
            observations, actions, rewards, next_observations = self.replayer.sample(self.batch_size)

            observations = observations.reshape(-1, self.width, self.height, self.frame_count)
            actions = actions.astype(np.int8)
            next_observations = next_observations.reshape(-1, self.width, self.height, self.frame_count)

            # 计算回报的估计值
            q_next = self.target_net.predict(next_observations)
            q_target = self.evaluate_net.predict(observations)
            q_target[range(self.batch_size), actions] = rewards + self.gamma * q_next.max(axis=-1)

            self.evaluate_net.fit(observations, q_target, verbose=0)

            self.save_evaluate_network()

        self.step += 1

    # 更新目标网络权重方法
    def update_target_network(self):
        self.target_net.set_weights(self.evaluate_net.get_weights())

    # 保存评估网络权重方法
    def save_evaluate_network(self):
        self.evaluate_net.save_weights(self.save_weights_path)