import os
import time

import numpy as np

from pysekiro.Agent import RewardSystem
from pysekiro.get_keys import key_check
from pysekiro.get_status import get_status
from pysekiro.grab_screen import get_screen

# 根据 get_output 函数来定义
action_map = {
    0: 'Attack',
    1: 'Deflect',
    2: 'Step Dodge',
    3: 'Jump',
    4: 'O'    # Other
}

def get_output(keys):    # 对按键信息进行独热编码

    output = [0, 0, 0, 0, 0]

    if   'J' in keys:
        output[0] = 1    # 等同于[1, 0, 0, 0, 0]
    elif 'K' in keys:
        output[1] = 1    # 等同于[0, 1, 0, 0, 0]
    elif 'LSHIFT' in keys:
        output[2] = 1    # 等同于[0, 0, 1, 0, 0]
    elif 'SPACE' in keys:
        output[3] = 1    # 等同于[0, 0, 0, 1, 0]
    else:
        output[4] = 1    # 等同于[0, 0, 0, 0, 1]

    return output

class Data_collection:
    def __init__(self, target):
        self.target = target    # 目标
        self.dataset = list()    # 保存数据的容器
        self.save_path = os.path.join('The_battle_memory', self.target)    # 保存的位置
        if not os.path.exists(self.save_path):    # 确保保存的位置存在
            os.mkdir(self.save_path)
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        self.reward_system = RewardSystem()    # 奖惩系统

        self.step = 0    # 计步器

    def save_data(self):
        print('\n\nStop, please wait')
        n = 1
        while True:    # 直到找到保存位置并保存就 break
            filename = f'training_data-{n}.npy'
            save_path = os.path.join(self.save_path, filename)
            if not os.path.exists(save_path):    # 没有重复的文件名就执行保存并退出
                print(save_path)
                np.save(save_path, self.dataset)
                break
            n += 1
        print('Done!')
        return filename[:-4]

    def collect_data(self):

        print('Ready!')
        paused = True
        while True:
            last_time = time.time()
            keys = key_check()
            if paused:
                if 'T' in keys:
                    paused = False
                    print('Starting!')
            else:
                
                self.step += 1

                screen = get_screen()    # 获取屏幕图像
                if not (np.sum(screen == 0) > 5000):    # 正常情况下不会有那么多值为0的像素点，除非黑屏了
                    action = get_output(keys)    # 获取按键输出
                    self.dataset.append([screen, action])    # 图像和输出打包在一起，保证一一对应

                    status = get_status(screen)
                    Self_HP, Self_Posture, Target_HP, Target_Posture = status
                    reward = self.reward_system.get_reward(status)    # 计算 reward

                    # 降低数据采集的频率，两次采集的时间间隔为0.1秒
                    t = 0.1-(time.time()-last_time)
                    if t > 0:
                        time.sleep(t)

                    print(f'\rstep:{self.step:>4}. Loop took {round(time.time()-last_time, 3):>5} seconds. \
                        action: {action_map[np.argmax(action)]:>10}. \
                        Self HP: {Self_HP:>3}, Self Posture: {Self_Posture:>3}, Target HP: {Target_HP:>3}, Target Posture: {Target_Posture:>3}', end='')

                if 'P' in keys:    # 结束，保存数据
                    filename = self.save_data()    # 保存数据，保存结束后返回符合条件的文件名
                    self.reward_system.save_reward_curve(
                        save_path = os.path.join('Data_quality', self.target, filename+'.png')
                    )    # 绘制 reward 曲线并保存在当前目录
                    break