import os
import threading
import time

import cv2
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import pandas as pd

from pysekiro.Agent import Sekiro_Agent
from pysekiro.key_tools.get_keys import key_check
from pysekiro.img_tools.get_status import get_status
from pysekiro.img_tools.get_vertices import roi
from pysekiro.img_tools.grab_screen import get_screen

# ---*---

WIDTH   = 300
HEIGHT  = 300
FRAME_COUNT = 1
n_action = 5

x   = 250
x_w = 550
y   = 75
y_h = 375

# ---*---

class Play_Sekiro:
    def __init__(
        self,
        load_weights_path=None,    # 指定模型权重参数加载的路径。默认为None，不加载。
        save_weights_path=None,    # 指定模型权重参数保存的路径。默认为None，不保存。
        load_memory_path=None,     # 指定记忆加载的路径。默认为None，不加载。
        save_memory_path=None,     # 指定记忆保存的路径。默认为None，不保存。
    ):
        self.sekiro_agent = Sekiro_Agent(
            width = WIDTH,
            height = HEIGHT,
            frame_count = FRAME_COUNT,
            n_action = n_action,
            learning_rate = 0.01,
            load_weights_path = load_weights_path,
            save_weights_path = save_weights_path
        )

        if not save_weights_path:    # 注：默认也是测试模式，若设置该参数，就会开启训练模式
            self.train = False
        else:
            self.train = True

        self.load_memory_path = load_memory_path
        self.save_memory_path = save_memory_path

        self.step = 1    # 计步器

        self.prepare()    # 初次准备

        self.Continue = True
        self.thread = threading.Thread(target=self.get_screen_continuously)
        self.thread.start()    # 开始不断的抓取屏幕获取信息

    def prepare(self):
        # 初次训练需要加载GPU，这个过程有那么一点点长，初次训练之后就不会加载那么长时间了，所以这里先训练一次，避免影响到后面正式的训练
        self.sekiro_agent.evaluate_net.fit(
            np.zeros((HEIGHT, WIDTH, FRAME_COUNT)).reshape(-1, WIDTH, HEIGHT, FRAME_COUNT),
            np.array([[0, 0, 0, 0, 1]]),
            verbose=0
        )

        # 记忆存在就读取记忆
        if self.load_memory_path != None:
            
            if os.path.exists(self.load_memory_path):
                
                self.sekiro_agent.replayer.memory = pd.read_json(self.load_memory_path)
                print('Load ' + self.load_memory_path)

                i = self.sekiro_agent.replayer.memory.action.count()
                self.sekiro_agent.replayer.i = i
                self.sekiro_agent.replayer.count = i

                # 恢复探索率
                self.sekiro_agent.epsilon -= self.sekiro_agent.epsilon_decrease_rate * i

                # 恢复步数，但去除零头，加1是为了避免刚进入就马上训练
                self.sekiro_agent.step = i // self.sekiro_agent.target_network_update_freq * self.sekiro_agent.target_network_update_freq + 1

            else:
                print('No memory to load.')

    def get_screen_continuously(self):
        
        while self.Continue:
            
            self.screen = get_screen()
            
            status = get_status(self.screen)
            self.status = status[:4]        # 生命架势状态
            self.status_info = status[4]    # 生命架势状态信息
            
            self.observation = cv2.resize(roi(self.screen, x, x_w, y, y_h), (WIDTH, HEIGHT))

    def run(self):

        paused = True
        print("Ready!")

        while True:

            self.last_time = time.time()
            
            keys = key_check()
            
            if paused:
                if 'T' in keys:
                    self.sekiro_agent.reward_system.cur_status = self.status[:4]    # 设置初始状态
                    paused = False
                    print('\nStarting!')

            else:    # 按 'T' 之后，马上进入下一轮就进入这里

                self.step += 1

                self.getSARS_()

                print(f'\r {self.sekiro_agent.who_play:>4} , step: {self.step:>6} . Loop took {round(time.time()-self.last_time, 3):>5} seconds. action {self.action:>1} , {self.status_info} , total_reward: {self.sekiro_agent.reward_system.total_reward:>10.3f} , memory: {self.sekiro_agent.replayer.count:7>} .', end='')
 
                if 'P' in keys:
                    self.Continue = False    # 结束抓取屏幕获取信息
                    if self.train:
                        self.sekiro_agent.save_evaluate_network()    # 学习完毕，保存网络权重
                        if self.save_memory_path:
                            self.sekiro_agent.replayer.memory.to_json(self.save_memory_path)    # 保存记忆
                    self.sekiro_agent.reward_system.save_reward_curve()    # 绘制 reward 曲线并保存在当前目录
                    break

        print('\nDone!')

    def getSARS_(self):

        # 1. 获取 状态S
        # 2. 由 状态S 选取 动作A
        # 3. 等待一段时间
        # 5. 获取 新状态S'，并计算 奖励R

        observation = self.observation    # 状态S

        action = self.action = self.sekiro_agent.choose_action(observation, self.train)    # 选取 动作A

        # 延迟观测新状态，为了能够观测到状态变化
        time.sleep(0.2)

        reward = self.sekiro_agent.reward_system.get_reward(self.status)    # 计算 奖励R

        next_observation = self.observation    # 新状态S'

        self.learn(observation, action, reward, next_observation)

    def learn(self, observation, action, reward, next_observation):
        if self.train:
            if not (np.sum(observation == 0) > int(WIDTH * HEIGHT * FRAME_COUNT / 8)):    # 当图像有1/8变成黑色（像素值为0）的时候停止暂停存储数据

                # ----- store -----
                self.sekiro_agent.replayer.store(
                    observation,
                    action,
                    reward,
                    next_observation
                )    # 存储经验

                # ----- learn -----
                self.sekiro_agent.learn()