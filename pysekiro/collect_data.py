import os
import time

import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

from pysekiro.get_keys import key_check
from pysekiro.grab_screen import get_screen

def get_output(keys):    # 对按键信息进行独热编码

    output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # 攻击、防御、垫步、跳跃和使用道具不能同时进行（指0.1秒内），但是可以和移动同时进行
    if   'J' in keys:
        output[0] = 1    # 等同于[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif 'K' in keys:
        output[1] = 1    # 等同于[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif 'LSHIFT' in keys:
        output[2] = 1    # 等同于[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif 'SPACE' in keys:
        output[3] = 1    # 等同于[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif 'R' in keys:
        output[5] = 1    # 等同于[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    else:
        output[4] = 1    # 等同于[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    # 不能同时前后移动
    if   'W' in keys:
        output[6] = 1    # 等同于[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif 'S' in keys:
        output[7] = 1    # 等同于[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

    # 不能同时左右移动
    if   'A' in keys:
        output[8] = 1    # 等同于[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif 'D' in keys:
        output[9] = 1    # 等同于[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    return output

class Data_collection:
    def __init__(self, target):
        self.target = target    # 目标
        self.dataset = list()    # 保存数据的容器
        self.save_path = os.path.join('The_battle_memory', self.target)    # 保存的位置
        if not os.path.exists(self.save_path):    # 确保保存的位置存在
            os.mkdir(self.save_path)

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
                if not (np.sum(screen == 0) > 32400):    # 270 * 480 / 4 = 32400 ，当图像有1/4变成黑色（像素值为0）的时候停止暂停数据数据
                    action_onehot = get_output(keys)    # 获取按键输出
                    self.dataset.append([screen, action_onehot])    # 图像和输出打包在一起，保证一一对应

                # 降低数据采集的频率，两次采集的时间间隔为0.1秒
                t = 0.1-(time.time()-last_time)
                if t > 0:
                    time.sleep(t)

                print(f'\rstep:{self.step:>4}. Loop took {round(time.time()-last_time, 3):>5} seconds.', end='')

                if 'P' in keys:    # 结束，保存数据
                    self.save_data()    # 保存数据
                    break