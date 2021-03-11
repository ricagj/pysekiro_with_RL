import time

import cv2
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

from pysekiro.Agent import Sekiro_Agent
from pysekiro.get_keys import key_check
from pysekiro.get_status import get_status
from pysekiro.get_vertices import roi
from pysekiro.grab_screen import get_screen

# ---*---

ROI_WIDTH   = 100
ROI_HEIGHT  = 100

x   = 140
x_w = 340
y   = 30
y_h = 230

# ---*---

# 在线学习
def learn_online(
    model_weights=None,
    save_path=None,
    reward_curve_save_path='learn_online.png'
    ):

    sekiro_agent = Sekiro_Agent(
        model_weights=model_weights,
        save_path = save_path
    )
    
    if save_path:    # 判断是训练模式还是测试模式
        train = True
    else:
        train = False

    paused = True
    print("Ready!")

    step = 0    # 计步器

    while True:

        last_time = time.time()
        keys = key_check()
        if paused:
            screen = get_screen()    # 首个 状态S，但是在按 'T' 之前，它会不断更新
            sekiro_agent.reward_system.cur_status = get_status(screen)    # 设置初始状态
            if 'T' in keys:
                paused = False
                print('\nStarting!')
        else:

            step += 1

            # ---------- (S, A, R, S') ----------

            action = sekiro_agent.choose_action(screen, train)    # 动作A
            next_screen = get_screen()    # 新状态S'
            reward = sekiro_agent.reward_system.get_reward(get_status(next_screen))    # 奖励R

            # ---------- 下一个轮回 ----------

            screen = next_screen    # 状态S

            if train:
                if not (np.sum(screen == 0) > 97200):    # 270 * 480 * 3 / 4 = 97200 ，当图像有1/4变成黑色（像素值为0）的时候停止暂停存储数据

                    # ---------- store ----------

                    # 集齐 (S, A, R, S')，开始存储
                    sekiro_agent.replayer.store(
                        cv2.resize(roi(screen, x, x_w, y, y_h), (ROI_WIDTH, ROI_HEIGHT)),    # 截取感兴趣区域并图像缩放
                        action,
                        reward,
                        cv2.resize(roi(next_screen, x, x_w, y, y_h), (ROI_WIDTH, ROI_HEIGHT))    # 截取感兴趣区域并图像缩放
                    )    # 存储经验

                    # ---------- learn ----------

                    sekiro_agent.step += 1
                    sekiro_agent.learn()

            # 降低数据采集的频率，两次采集的时间间隔为0.1秒
            t = 0.1-(time.time()-last_time)
            if t > 0:
                time.sleep(t)

            print(f'\rstep:{step:>4}. Loop took {round(time.time()-last_time, 3):>5} seconds. action {action}', end='')
            
            if 'P' in keys:
                if train:
                    sekiro_agent.save_evaluate_network()    # 学习完毕，保存网络权重
                sekiro_agent.reward_system.save_reward_curve(save_path=reward_curve_save_path)    # 绘制 reward 曲线并保存在当前目录
                break

    print('\nDone!')