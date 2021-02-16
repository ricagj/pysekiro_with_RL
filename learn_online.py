import time

import numpy as np

from pysekiro.Agent import Sekiro_Agent
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
update_freq = 200
# 更新目标网络的频率
target_network_update_freq = 500

action_weight = [1.0, 2.0, 5.0, 5.0, 0.1]

# ---*---

# 在线学习
def learn_online(train=False):

    sekiro_agent = Sekiro_Agent(action_weight=action_weight)

    paused = True
    print("Ready!")

    step = 0    # 初始化计数值

    while True:

        if paused:

            # 暂停状态启用，保证 screen 和 next_screen 连续
            screen = get_screen()

        else:
            last_time = time.time()

            # 选取动作，同时执行动作
            action = sekiro_agent.choose_action(screen)

            next_screen = get_screen()

            status = get_status(screen)
            sekiro_agent.reward_system.store(status)    # 存储状态 [我方生命, 我方架势, 敌方生命, 敌方架势]
            reward = sekiro_agent.reward_system.get_reward()    # 计算 reward

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
                        print(f'step:{step:>5}, current_cumulative_reward:{sekiro_agent.reward_system.current_cumulative_reward:>5.3f}, memory:{sekiro_agent.replayer.count:7>}')
                        sekiro_agent.update_target_network()

            screen = next_screen

            print(f'Loop took {round(time.time()-last_time, 3):>5} seconds.')

        keys = key_check()
        # 优先检测终止指令，再检测暂停指令
        if 'P' in keys:
            if train:
                sekiro_agent.save_evaluate_network()    # 学习完毕，保存网络权重
            sekiro_agent.reward_system.save_reward_curve(save_path='.\\test.png')    # 绘制 reward 曲线并保存在当前目录
            break

        # 准备切换暂停状态
        elif 'T' in keys:
            if paused:
                paused = False
                print('\nStarting!')
                time.sleep(1)
            else:
                paused = True
                print('\nPausing!')
                time.sleep(1)

    print('\nDone!')

# ---*---

learn_online(train=True)