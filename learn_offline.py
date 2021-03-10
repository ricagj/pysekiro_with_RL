import os

import cv2
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

from pysekiro.Agent import Sekiro_Agent
from pysekiro.get_status import get_status
from pysekiro.get_vertices import roi

# ---*---

ROI_WIDTH   = 100
ROI_HEIGHT  = 100

x   = 140
x_w = 340
y   = 30
y_h = 230

# ---*---

# 离线学习
def learn_offline(
    target,
    start=1,
    end=1,
    model_weights=None,
    save_path=None,
    reward_curve_save_path='learn_offline.png'
    ):

    sekiro_agent = Sekiro_Agent(
        model_weights = model_weights,
        save_path = save_path
    )

    # 依次读取训练集进行离线学习
    for i in range(start, end+1):

        filename = f'training_data-{i}.npy'
        data_path = os.path.join('The_battle_memory', target, filename)
        
        if os.path.exists(data_path):    # 确保数据集存在

            # 加载数据集
            data = np.load(data_path, allow_pickle=True)
            print('\n', filename, f'total:{len(data):>5}')
            
            sekiro_agent.reward_system.cur_status = get_status(dataset[0][0])    # 设置初始状态
            for step in range(len(data)-1):

                # ---------- (S, A, R, S') ----------
                
                screen = data[step][0]               # 状态S
                action = np.argmax(data[step][1][:5])    # 动作A
                next_screen = data[step+1][0]        # 新状态S'
                reward = sekiro_agent.reward_system.get_reward(get_status(next_screen))    # 奖励R

                # ---------- store ----------
                
                # 集齐 (S, A, R, S')，开始存储
                sekiro_agent.replayer.store(
                    cv2.resize(roi(screen, x, x_w, y, y_h), (ROI_WIDTH, ROI_HEIGHT)),    # 截取感兴趣区域并图像缩放
                    action,
                    reward,
                    cv2.resize(roi(next_screen, x, x_w, y, y_h), (ROI_WIDTH, ROI_HEIGHT))    # 截取感兴趣区域并图像缩放
                )    # 存储经验

                # ---------- learn ----------
                
                sekiro_agent.step = step + 1
                sekiro_agent.learn()

            sekiro_agent.save_evaluate_network()    # 这个数据学习完毕，保存网络权重
            sekiro_agent.reward_system.save_reward_curve(save_path=reward_curve_save_path)    # 绘制 reward 曲线并保存
            print(f'[summary] round:{i:>3}, current_cumulative_reward:{sekiro_agent.reward_system.current_cumulative_reward:>5.3f}, memory:{sekiro_agent.replayer.count:7>}', end='\n\n')

        else:
            print(f'{filename} does not exist ')