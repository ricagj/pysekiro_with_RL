import time

import cv2
import numpy as np

from pysekiro.Agent import Sekiro_Agent
from pysekiro.collect_data import action_map
from pysekiro.get_keys import key_check
from pysekiro.get_status import get_status
from pysekiro.get_vertices import roi
from pysekiro.grab_screen import get_screen

# ---*---

x   = 190
x_w = 290
y   = 30
y_h = 230

# 训练评估网络的频率
update_freq = 150
# 更新目标网络的频率
target_network_update_freq = 900

# ---*---

# 在线学习
def learn_online(model_weights=None, save_path=None, train=False):

    sekiro_agent = Sekiro_Agent(
        model_weights=model_weights,
        save_path = save_path
    )

    paused = True
    print("Ready!")

    step = 0    # 初始化计数值

    while True:

        last_time = time.time()
        keys = key_check()
        
        if paused:
            screen = get_screen()
            if 'T' in keys:
                paused = False
                print('\nStarting!')
        else:

            # 选取动作，同时执行动作
            action = sekiro_agent.choose_action(screen, train)

            status = get_status(screen)
            Self_HP, Self_Posture, Target_HP, Target_Posture = status
            reward = sekiro_agent.reward_system.get_reward(status)    # 计算 reward

            next_screen = get_screen()
            if train:
                sekiro_agent.replayer.store(
                    roi(screen, x, x_w, y, y_h),
                    np.argmax(action),
                    reward,
                    roi(next_screen, x, x_w, y, y_h)
                )    # 存储经验

                step += 1

                if step >= sekiro_agent.batch_size:
                    if step % update_freq == 0:
                        sekiro_agent.learn()
                        sekiro_agent.save_evaluate_network()

                    if step % target_network_update_freq == 0:
                        print(f'\n step:{step:>5}, current_cumulative_reward:{sekiro_agent.reward_system.current_cumulative_reward:>5.3f}, memory:{sekiro_agent.replayer.count:7>} \n')
                        sekiro_agent.update_target_network()

            screen = next_screen

            print(f'\rloop took {round(time.time()-last_time, 3):>5} seconds. action: {action_map[np.argmax(action)]:>10}. Self HP: {Self_HP:>3}, Self Posture: {Self_Posture:>3}, Target HP: {Target_HP:>3}, Target Posture: {Target_Posture:>3}', end='')
            
            
            if 'P' in keys:
                if train:
                    sekiro_agent.save_evaluate_network()    # 学习完毕，保存网络权重
                sekiro_agent.reward_system.save_reward_curve(save_path='learn_online.png')    # 绘制 reward 曲线并保存在当前目录
                break

    print('\nDone!')