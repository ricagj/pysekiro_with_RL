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
update_freq = 200
# 更新目标网络的频率
target_network_update_freq = 500



# ---*---

# 离线学习
def learn_offline(boss, start, end):

    sekiro_agent = Sekiro_Agent()

    for i in range(start, end+1):

        filename = f'training_data-{i}.npy'
        data = np.load(os.path.join('The_battle_memory', boss, filename), allow_pickle=True)
        print('\n', filename, f'total:{len(data):>5}')

        for step in range(len(data)-1):

            screen = data[step][0]
            action = data[step][1]
            next_screen = data[step+1][0]

            status = get_status(screen)
            sekiro_agent.reward_system.store(status)    # 存储状态 [我方生命, 我方架势, 敌方生命, 敌方架势]
            reward = sekiro_agent.reward_system.get_reward()    # 计算 reward

            sekiro_agent.replayer.store(
                roi(screen, x, x_w, y, y_h),
                np.argmax(action),
                reward,
                roi(next_screen, x, x_w, y, y_h)
            )    # 存储经验

            if step >= sekiro_agent.batch_size:
                if step % update_freq == 0:
                    sekiro_agent.learn()
                    sekiro_agent.save_evaluate_network()

                if step % target_network_update_freq == 0:
                    print(f'step:{step:>5}')
                    sekiro_agent.update_target_network()

        sekiro_agent.save_evaluate_network()    # 这个数据学习完毕，保存网络权重
        sekiro_agent.reward_system.save_reward_curve()    # 绘制 reward 曲线并保存在当前目录
        print(f'[summary] round:{i:>3}, current_cumulative_reward:{sekiro_agent.reward_system.current_cumulative_reward:>5.3f}, memory:{sekiro_agent.replayer.count:7>}')

# ---*---

boss = 'Genichiro_Ashina' # 苇名弦一郎
learn_offline(boss, start=1, end=101)