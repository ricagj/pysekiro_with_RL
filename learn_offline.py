import os

import numpy as np

from pysekiro.Agent import Sekiro_Agent
from pysekiro.get_status import get_status
from pysekiro.get_vertices import roi

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

# 离线学习
def learn_offline(target, start, end, model_weights=None, save_path=None):

    sekiro_agent = Sekiro_Agent(
        batch_size = 128,
        replay_memory_size = 200000,
        model_weights=model_weights,
        save_path = save_path
    )

    for i in range(start, end+1):

        filename = f'training_data-{i}.npy'
        data = np.load(os.path.join('The_battle_memory', target, filename), allow_pickle=True)
        print('\n', filename, f'total:{len(data):>5}')

        for step in range(len(data)-1):

            # 读取 状态S、动作A 和 新状态S'
            screen = data[step][0]           # 状态S
            action = data[step][1]           # 动作A
            next_screen = data[step+1][0]    # 新状态S'

            # 获取 奖励R
            status = get_status(screen)
            reward = sekiro_agent.reward_system.get_reward(status)    # 奖励R

            # 集齐 (S, A, R, S')，开始存储
            sekiro_agent.replayer.store(
                roi(screen, x, x_w, y, y_h),
                np.argmax(action),
                reward,
                roi(next_screen, x, x_w, y, y_h)
            )    # 存储经验

            if step >= sekiro_agent.batch_size:
                if step % update_freq == 0:    # 更新评估网络
                    sekiro_agent.learn()
                    sekiro_agent.save_evaluate_network()

                if step % target_network_update_freq == 0:    # 更新目标网络
                    print(f'\r step:{step:>5}', end='')
                    sekiro_agent.update_target_network()

        sekiro_agent.save_evaluate_network()    # 这个数据学习完毕，保存网络权重
        sekiro_agent.reward_system.save_reward_curve()    # 绘制 reward 曲线并保存在当前目录
        print(f'\r [summary] round:{i:>3}, current_cumulative_reward:{sekiro_agent.reward_system.current_cumulative_reward:>5.3f}, memory:{sekiro_agent.replayer.count:7>}')