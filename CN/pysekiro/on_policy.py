from collections import deque
import os
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
        if sum(next_status) == 0:    # [0, 0, 0, 0]

            reward = 0

        else:

            s1 = min(0, next_status[0] - cur_status[0]) *  1    # 自身生命，忽略增加
            t1 = min(0, next_status[2] - cur_status[2]) * -1    # 目标生命，忽略增加
            s2 = max(0, next_status[1] - cur_status[1]) * -1    # 自身架势，忽略减少
            t2 = max(0, next_status[3] - cur_status[3]) *  1    # 目标架势，忽略减少

            reward = s1 + s2 + t1 + t2

        Self_HP = next_status[0]
        if Self_HP < 3:
            Reset_Self_HP()    # 重置自身生命值。注：先把修改器开着，不然这一步无效
            time.sleep(1)
            Lock_On()    # 重置视角/固定目标

            reward = -300

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

x   = 250
x_w = 550
y   = 75
y_h = 375

in_depth    = 10
in_height   = 50
in_width    = 50

# ---*---

class Play_Sekiro_Online:
    def __init__(
        self,
        save_memory_path=None,
        load_memory_path=None,
        save_weights_path=None,
        load_weights_path=None
    ):
        self.save_memory_path = save_memory_path     # 指定记忆/经验保存的路径。默认为None，不保存。
        self.load_memory_path = load_memory_path     # 指定记忆/经验加载的路径。默认为None，不加载。
        self.sekiro_agent = Sekiro_Agent(
            save_weights_path = save_weights_path,    # 指定模型权重保存的路径。默认为None，不保存。
            load_weights_path = load_weights_path     # 指定模型权重加载的路径。默认为None，不加载。
        )
        if not save_weights_path:    # 注：默认也是测试模式，若设置该参数，就会开启训练模式
            self.train = False
            self.sekiro_agent.step = self.sekiro_agent.replay_start_size + 1
        else:
            self.train = True

        self.reward_system = RewardSystem()

        self.i = 0    # 计步器

        self.screens = deque(maxlen = in_depth * 2)    # 用双端队列存放图像

        if self.load_memory_path:
            self.load_memory()    # 加载记忆/经验

    def load_memory(self):
        if os.path.exists(self.load_memory_path):
            last_time = time.time()
            self.sekiro_agent.replayer.memory = pd.read_json(self.load_memory_path)    # 从json文件加载记忆/经验。 
            print(f'Load {self.load_memory_path}. Took {round(time.time()-last_time, 3):>5} seconds.')

            i = self.sekiro_agent.replayer.memory.action.count()
            self.sekiro_agent.replayer.i = i
            self.sekiro_agent.replayer.count = i
            self.sekiro_agent.step = i

        else:
            print('No memory to load.')

    def get_S(self):

        for _ in range(in_depth):
            self.screens.append(get_screen())    # 先进先出，右进左出

    def img_processing(self, screens):
        return np.array([cv2.resize(roi(cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY), x, x_w, y, y_h), (in_height, in_width)) for screen in screens])

    def round(self):

        observation = self.img_processing(list(self.screens)[:in_depth])    # S

        action = self.action = self.sekiro_agent.choose_action(observation)    # A

        self.get_S()

        reward = self.reward_system.get_reward(
            cur_status=get_status(list(self.screens)[in_depth - 1])[:4],
            next_status=get_status(list(self.screens)[in_depth * 2 - 1])[:4]
        )    # R

        next_observation = self.img_processing(list(self.screens)[in_depth:])    # S'

        if self.train:

            self.sekiro_agent.replayer.store(
                observation,
                action,
                reward,
                next_observation
            )

            if self.sekiro_agent.replayer.count >= self.sekiro_agent.replay_start_size:
                self.sekiro_agent.learn()

    def run(self):

        paused = True
        print("Ready!")

        while True:

            last_time = time.time()
            
            keys = key_check()
            
            if paused:
                if 'T' in keys:
                    self.get_S()
                    paused = False
                    print('\nStarting!')

            else:    # 按 'T' 之后，马上下一轮就进入这里

                self.i += 1

                self.round()

                print(f'\r {self.sekiro_agent.who_play:>4} , step: {self.i:>6} . Loop took {round(time.time()-last_time, 3):>5} seconds. action {self.action:>1} , total_reward: {self.reward_system.total_reward:>10.3f} , memory: {self.sekiro_agent.replayer.count:7>} .', end='')
 
                if 'P' in keys:
                    if self.train:
                        self.sekiro_agent.save_evaluate_network()    # 学习完毕，保存网络权重
                        self.sekiro_agent.replayer.memory.to_json(self.save_memory_path)    # 保存经验
                    self.reward_system.save_reward_curve()    # 绘制 reward 曲线并保存在当前目录
                    break

        print('\nDone!')