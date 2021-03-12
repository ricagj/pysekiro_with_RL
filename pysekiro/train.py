import time

import cv2
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

from pysekiro.Agent import Sekiro_Agent
from pysekiro.key_tools.get_keys import key_check
from pysekiro.img_tools.get_status import get_status
from pysekiro.img_tools.get_vertices import roi
from pysekiro.img_tools.grab_screen import get_screen

# ---*---

RESIZE_WIDTH = 100
RESIZE_HEIGHT = 100

x   = 390
x_w = 890
y   = 110
y_h = 610

n_action = 5

# ---*---

class Play_Sekiro:
    def __init__(
        self,
        batch_size=16,    # 样本抽取数量
        model_weights=None,
        save_path=None,
        reward_curve_save_path='reward.png',
        cheating_mode=False,
        play_yourself=False
    ):
        self.sekiro_agent = Sekiro_Agent(n_action, batch_size, model_weights, save_path)
        if save_path:    # 判断是训练模式还是测试模式
            self.train = True
        else:
            self.train = False
        self.reward_curve_save_path = reward_curve_save_path
        self.cheating_mode = cheating_mode
        self.step = 1    # 计步器

    def learn(self):
        if self.train:
            if not (np.sum(self.screen == 0) > 7500):    # 100 * 100 * 3 / 4 = 7500 ，当图像有1/4变成黑色（像素值为0）的时候停止暂停存储数据

                # ----- store -----
                # 集齐 (S, A, R, S')后开始存储
                self.sekiro_agent.replayer.store(
                    self.screen,    # 截取感兴趣区域并图像缩放
                    self.action,
                    self.reward,
                    self.next_screen    # 截取感兴趣区域并图像缩放
                )    # 存储经验

                # ----- learn -----
                self.sekiro_agent.step += 1
                self.sekiro_agent.learn()

    def getSARS_(self):

        # 第二个轮回开始
        #   1. 原本的 新状态S' 变成 状态S
        #   2. 由 状态S 选取 动作A
        #   3. 观测并获取 新状态S'，并计算 奖励R
        # 进入下一个轮回

        self.action = self.sekiro_agent.choose_action(self.screen, self.train)    # 选取 动作A
        next_screen = get_screen()    # 观测 新状态
        self.reward = self.sekiro_agent.reward_system.get_reward(get_status(next_screen), self.cheating_mode)    # 计算 奖励R
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
                screen = get_screen()
                self.sekiro_agent.reward_system.cur_status = get_status(screen)    # 设置初始状态
                self.screen = cv2.resize(roi(screen, x, x_w, y, y_h), (RESIZE_WIDTH, RESIZE_HEIGHT))    # 首个 状态S，但是在按 'T' 之前，它会不断更新
                if 'T' in keys:    # 按 'T' 之后，开始训练，此时刚刚获得的 状态S 就是首个 状态S，然后进入 getSARS_() 依次获得 动作A、奖励R 和 新状态S'，再进入第二个轮回  
                    paused = False
                    print('\nStarting!')
            else:

                self.step += 1

                self.getSARS_()

                # 降低数据采集的频率，两次采集的时间间隔为0.1秒
                t = 0.1-(time.time()-last_time)
                if t > 0:
                    time.sleep(t)

                print(f'\rstep:{self.step:>5}. Loop took {round(time.time()-last_time, 3):>5} seconds. action {self.action:>2}', end='')

                if 'P' in keys:
                    if self.train:
                        self.sekiro_agent.save_evaluate_network()    # 学习完毕，保存网络权重
                    self.sekiro_agent.reward_system.save_reward_curve(save_path=self.reward_curve_save_path)    # 绘制 reward 曲线并保存在当前目录
                    break

        print('\nDone!')