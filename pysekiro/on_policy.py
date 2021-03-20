from collections import deque
import os
import threading
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import pandas as pd

from pysekiro.Agent import Sekiro_Agent
from pysekiro.img_tools.get_status import get_status
from pysekiro.img_tools.get_vertices import roi
from pysekiro.img_tools.grab_screen import get_screen
from pysekiro.key_tools.actions import Lock_On, Reset_Self_HP
from pysekiro.key_tools.get_keys import key_check

# ---*---

class RewardSystem:
    def __init__(self):
        self.total_reward = 0    # 当前积累的 reward
        self.reward_history = list()    # reward 的积累过程

    # 获取奖励
    def get_reward(self, cur_status, next_status):
        if sum(next_status) == 0:    # 状态全为0，即出现异常

            reward = 0

        else:

            # 计算方法：(下一个的状态 - 当前的状态) * 各自的正负强化权重
            s1 = min(0, next_status[0] - cur_status[0]) *  1    # 自身生命，不计增加
            t1 = min(0, next_status[2] - cur_status[2]) * -1    # 目标生命，不计增加
            s2 = max(0, next_status[1] - cur_status[1]) * -1    # 自身架势，不计减少
            t2 = max(0, next_status[3] - cur_status[3]) *  1    # 目标架势，不计减少

            reward = s1 + s2 + t1 + t2

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

x   = 200
x_w = 600
y   = 25
y_h = 425

in_depth    = 8
in_depth    = in_depth // 2 * 2
in_height   = (y_h - y) // 4
in_width    = (x_w - x) // 4
in_channels = 1
outputs     = 5
lr          = 0.01

min_epsilon = 0.3
replay_memory_size = in_depth * 1000
replay_start_size  = in_depth * 50
batch_size = 32 // in_depth
update_freq = in_depth * 20
target_network_update_freq = in_depth * 100

# ---*---

class Play_Sekiro_Online:
    def __init__(
        self,
        load_weights_path=None,    # 指定模型权重参数加载的路径。默认为None，不加载。
        save_weights_path=None,    # 指定模型权重参数保存的路径。默认为None，不保存。
        load_memory_path=None,     # 指定记忆加载的路径。默认为None，不加载。
        save_memory_path=None,     # 指定记忆保存的路径。默认为None，不保存。
    ):
        self.sekiro_agent = Sekiro_Agent(
            in_depth    = in_depth,
            in_height   = in_height,
            in_width    = in_width,
            in_channels = in_channels,
            outputs     = outputs,
            lr          = lr,

            min_epsilon = min_epsilon,
            replay_memory_size = replay_memory_size,
            replay_start_size = replay_start_size,
            batch_size = batch_size,
            update_freq = update_freq,
            target_network_update_freq = target_network_update_freq,

            load_weights_path = load_weights_path,
            save_weights_path = save_weights_path
        )

        if not save_weights_path:    # 注：默认也是测试模式，若设置该参数，就会开启训练模式
            self.train = False
        else:
            self.train = True

        self.load_memory_path = load_memory_path
        self.save_memory_path = save_memory_path

        self.reward_system = RewardSystem()    # 奖惩系统

        self.i = 0    # 计步

        self.screens = deque(maxlen = in_depth * 2)    # 用双端队列存放图像

        if self.load_memory_path:
            self.load_memory()    # 加载经验

    def load_memory(self):
        if os.path.exists(self.load_memory_path):    # 确定经验的存在
            
            self.sekiro_agent.replayer.memory = pd.read_json(self.load_memory_path)    # 加载经验
            print('Load ' + self.load_memory_path)

            i = self.sekiro_agent.replayer.memory.action.count()    # 'observation', 'action', 'reward', 'next_observation' 都可以用，反正每一列数据的数量都一样
            self.sekiro_agent.replayer.i = i    # 恢复 index 指向的行索引（有点像指针）
            self.sekiro_agent.replayer.count = i    # 恢复 "代表经验的数量" 的数值

            # 恢复已学习的步数，但去除零头，加1是为了避免刚进入就马上训练
            self.sekiro_agent.step = i // self.update_freq * self.update_freq + 1

        else:
            print('No memory to load.')

    def get_S(self):

        for _ in range(in_depth):
            self.screens.append(get_screen())    # 右进左出

    def img_processing(self, screens):
        # 从内到外解释如何处理并得到observation
        # 颜色空间转换，BGR彩图转换成GRAY灰度图 cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        # 提取感兴趣区域 roi(？？？？？, x, x_w, y, y_h)
        # 图像缩放 cv2.resize(？？？？？, (in_height, in_width))
        # 从队列中取出刚刚获取的图像 [？？？？？ for screen in self.screens[:6]]
        return np.array([cv2.resize(roi(cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY), x, x_w, y, y_h), (in_height, in_width)) for screen in screens])

    def round(self):

        observation = self.img_processing(list(self.screens)[:in_depth])    # S

        action = self.action = self.sekiro_agent.choose_action(observation)    # A

        self.get_S()    # 获取新状态

        next_status = get_status(list(self.screens)[in_depth * 2 - 1])[:4]

        reward = self.reward_system.get_reward(
            cur_status=get_status(list(self.screens)[in_depth - 1])[:4],
            next_status=next_status
        )    # R

        Self_HP = next_status[0]
        if Self_HP < 5:    # 检测到生命值过低
            Reset_Self_HP()    # 重置自身生命值。注：先把修改器开着，不然这一步无效
            time.sleep(1)
            Lock_On()    # 如果已经凉了，还要重新锁定视角

            reward = -300    # 死亡惩罚

        next_observation = self.img_processing(list(self.screens)[in_depth:])    # S'

        if self.train:

            self.sekiro_agent.replayer.store(
                observation,
                action,
                reward,
                next_observation
            )    # 存储经验

            self.sekiro_agent.learn()

    def run(self):

        paused = True
        print("Ready!")

        while True:

            last_time = time.time()
            
            keys = key_check()
            
            if paused:
                if 'T' in keys:
                    self.get_S()    # 获取初状态
                    paused = False
                    print('\nStarting!')

            else:    # 按 'T' 之后，马上进入下一轮就进入这里

                self.i += 1

                self.round()

                print(f'\r {self.sekiro_agent.who_play:>4} , step: {self.i:>6} . Loop took {round(time.time()-last_time, 3):>5} seconds. action {self.action:>1} , total_reward: {self.reward_system.total_reward:>10.3f} , memory: {self.sekiro_agent.replayer.count:7>} .', end='')
 
                if 'P' in keys:
                    self.stop = True    # 停止获取信息
                    if self.train:
                        self.sekiro_agent.save_evaluate_network()    # 学习完毕，保存网络权重
                        self.sekiro_agent.replayer.memory.to_json(self.save_memory_path)    # 保存经验
                    self.reward_system.save_reward_curve()    # 绘制 reward 曲线并保存在当前目录
                    break

        print('\nDone!')