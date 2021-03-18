import os
import time

import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

from pysekiro.img_tools.grab_screen import get_screen
from pysekiro.key_tools.get_keys import key_check

WIDTH   = 100
HEIGHT  = 100
FRAME_COUNT = 3

def get_output(keys):

    output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # 对参与训练的按键信息进行独热编码
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
        output[5] = 1    # 等同于[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]    不参与训练
        output[4] = 1    # 等同于[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    else:
        output[4] = 1    # 等同于[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    # 不能同时前后移动
    if   'W' in keys:
        output[6] = 1    # 等同于[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]    不参与训练
    elif 'S' in keys:
        output[7] = 1    # 等同于[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]    不参与训练

    # 不能同时左右移动
    if   'A' in keys:
        output[8] = 1    # 等同于[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]    不参与训练
    elif 'D' in keys:
        output[9] = 1    # 等同于[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]    不参与训练

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
                if not (np.sum(screen == 0) > int(WIDTH * HEIGHT * FRAME_COUNT / 8)):    # 当图像有1/8变成黑色（像素值为0）的时候停止暂停存储数据
                    action_list = get_output(keys)    # 获取按键输出列表
                    self.dataset.append([screen, action_list])    # 图像和输出打包在一起，保证一一对应

                print(f'\rstep:{self.step:>4}. Loop took {round(time.time()-last_time, 3):>5} seconds.', end='')

                if 'P' in keys:    # 结束
                    self.save_data()    # 保存数据
                    break