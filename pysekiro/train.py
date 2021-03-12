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

RESIZE_WIDTH   = 50
RESIZE_HEIGHT  = 50
FRAME_COUNT    = 3

x   = 390
x_w = 890
y   = 110
y_h = 610

# ---*---

class Play_Sekiro:
    def __init__(
        self,
        load_weights_path=None,
        save_weights_path=None,
        load_memory_path=None,
        save_memory_path=None,
    ):
        self.sekiro_agent = Sekiro_Agent(
            load_weights_path = load_weights_path,
            save_weights_path = save_weights_path
        )

        # 初次训练需要加载GPU，这个过程有那么一点点长，初次训练之后就不用了，所以这里先训练一次，避免影响到后面正式的训练
        self.sekiro_agent.evaluate_net.fit(
            np.zeros((RESIZE_HEIGHT, RESIZE_WIDTH, FRAME_COUNT)).reshape(-1, RESIZE_WIDTH, RESIZE_HEIGHT, FRAME_COUNT),
            np.array([[0, 0, 0, 0, 1]]),
            verbose=0
        )

        if save_weights_path:    # 判断是训练模式还是测试模式
            self.train = True
        else:
            self.train = False
        
        self.save_memory_path = save_memory_path

        # 记忆存在就读取记忆
        if load_memory_path != None:
            if os.path.exists(load_memory_path):
                self.sekiro_agent.replayer.memory = pd.read_json(load_memory_path)
                i = self.sekiro_agent.replayer.memory.action.count()
                self.sekiro_agent.replayer.i = i
                self.sekiro_agent.replayer.count = i
            else:
                print('No memory to load.')

        self.step = 1    # 计步器

    def learn(self):
        if self.train:
            if not (np.sum(self.screen == 0) > 1875):    # 50 * 50 * 3 / 4 = 1875 ，当图像有1/4变成黑色（像素值为0）的时候停止暂停存储数据

                # ----- store -----
                # 集齐 (S, A, R, S')后开始存储
                self.sekiro_agent.replayer.store(
                    self.screen,    # 截取感兴趣区域并图像缩放
                    self.action,
                    self.reward,
                    self.next_screen    # 截取感兴趣区域并图像缩放
                )    # 存储经验

                # ----- learn -----
                self.sekiro_agent.learn()

    def getSARS_(self):

        # 第二个轮回开始
        #   1. 原本的 新状态S' 变成 状态S
        #   2. 由 状态S 选取 动作A
        #   3. 观测并获取 新状态S'，并计算 奖励R
        # 进入下一个轮回

        self.action = self.sekiro_agent.choose_action(self.screen, self.train)    # 选取 动作A
        next_screen = get_screen()    # 观测 新状态
        self.reward = self.sekiro_agent.reward_system.get_reward(get_status(next_screen))    # 计算 奖励R
        self.next_screen = cv2.resize(roi(next_screen, x, x_w, y, y_h), (RESIZE_WIDTH, RESIZE_HEIGHT))    # 获取 新状态S'

        self.learn()

        # ----- 下一个轮回 -----

        # 保证 状态S 和 新状态S' 连续
        self.screen = self.next_screen    # 状态S

    def run(self):

        paused = True
        print("Ready!")

        while True:

            last_time = time.time()
            keys = key_check()
            if paused:
                if 'T' in keys:
                    paused = False
                    print('\nStarting!')

                    screen = get_screen()
                    self.sekiro_agent.reward_system.cur_status = get_status(screen)    # 设置初始状态
                    self.screen = cv2.resize(roi(screen, x, x_w, y, y_h), (RESIZE_WIDTH, RESIZE_HEIGHT))    # 首个 状态S

            else:    # 按 'T' 之后，马上进入下一轮就进入这里

                self.step += 1

                self.getSARS_()

                # 降低数据采集的频率，两次采集的时间间隔为0.15秒
                t = 0.15-(time.time()-last_time)
                if t > 0:
                    time.sleep(t)

                print(f'\rstep:{self.step:>6}. Loop took {round(time.time()-last_time, 3):>5} seconds. action {self.action:>2}', end='')

                if 'P' in keys:
                    if self.train:
                        self.sekiro_agent.save_evaluate_network()    # 学习完毕，保存网络权重
                        if self.save_memory_path:
                            self.sekiro_agent.replayer.memory.to_json(self.save_memory_path)    # 保存记忆
                    self.sekiro_agent.reward_system.save_reward_curve()    # 绘制 reward 曲线并保存在当前目录
                    break

        print('\nDone!')