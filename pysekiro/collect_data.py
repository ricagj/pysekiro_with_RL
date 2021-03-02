import os
import time

import numpy as np

from pysekiro.Agent import RewardSystem
from pysekiro.get_keys import key_check
from pysekiro.get_status import get_status
from pysekiro.grab_screen import get_screen

# 根据 类Data_collection里的get_output方法
action_map = {
    0: 'Attack',
    1: 'Deflect',
    2: 'Step Dodge',
    3: 'Jump',
    4: 'O'    # Other
}

def get_output():    # 对按键信息进行独热编码

    keys = key_check()

    if   'J' in keys:
        output = [1,0,0,0,0]
    elif 'K' in keys:
        output = [0,1,0,0,0]
    elif 'LSHIFT' in keys:
        output = [0,0,1,0,0]
    elif 'SPACE' in keys:
        output = [0,0,0,1,0]
    else:
        output = [0,0,0,0,1]

    return output

class Data_collection:
    def __init__(
        self,
        target
    ):
        self.target = target
        self.dataset = list()    # 保存数据的容器
        self.save_path = os.path.join('The_battle_memory', self.target)    # 保存的位置
        if not os.path.exists(self.save_path):    # 确保保存的位置存在
            os.mkdir(self.save_path)
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        self.reward_system = RewardSystem()

    def save_data(self):
        print('\n\nStop, please wait')
        n = 1
        while True:
            save_path = os.path.join(self.save_path, f'training_data-{n}.npy')
            n += 1
            if not os.path.exists(save_path):    # 找到不会文件名重复的位置保存
                print(save_path)
                np.save(save_path, self.dataset)
                break
        print('Done!')

    def collect_data(self):

        print('Ready!')
        paused = True
        while True:
            if not paused:
                last_time = time.time()

                screen = get_screen()         # 获取屏幕图像
                action = get_output()    # 获取按键输出

                status = get_status(screen)
                Self_HP, Self_Posture, Target_HP, Target_Posture = status
                reward = self.reward_system.get_reward(status, np.argmax(action))    # 计算 reward

                self.dataset.append([screen, output])    # 图像和输出打包在一起，保证一一对应

                print(f'\rloop took {round(time.time()-last_time, 3):>5} seconds. action: {action_map[np.argmax(action)]:>10}. Self HP: {Self_HP:>3}, Self Posture: {Self_Posture:>3}, Target HP: {Target_HP:>3}, Target Posture: {Target_Posture:>3}', end='')

            keys = key_check()
            if 'P' in keys:    # 结束，保存数据
                self.save_data()
                self.reward_system.save_reward_curve(save_path='.\\collect_data_reward.png')    # 绘制 reward 曲线并保存在当前目录
                break
            elif 'T' in keys:    # 切换状态(暂停\继续)
                if paused:
                    paused = False
                    time.sleep(1)
                else:
                    paused = True
                    time.sleep(1)