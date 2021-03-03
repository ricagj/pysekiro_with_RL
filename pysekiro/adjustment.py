import time

import cv2
import numpy as np

from pysekiro.collect_data import get_output
from pysekiro.get_keys import key_check
from pysekiro.get_status import get_status
from pysekiro.grab_screen import get_screen

def main():

    paused = True
    print("Ready!")

    step = 0    # 初始化计数值

    while True:

        if not paused:
            last_time = time.time()

            screen = get_screen()    # 获取屏幕图像
            action = get_output()    # 获取按键输出

            status = get_status(screen)
            Self_HP, Self_Posture, Target_HP, Target_Posture = status

            screen[246:,[29, 182]] = 255
            screen[[233, 235],240:290] = 255
            screen[:25,[29, 130]] = 255
            screen[[16, 18],240:327] = 255
            cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
            cv2.imshow('screen', screen)
            cv2.waitKey(1)

            print(f'\rLoop took {round(time.time()-last_time, 3):>5} seconds. Self HP: {Self_HP:>3}, Self Posture: {Self_Posture:>3}, Target HP: {Target_HP:>3}, Target Posture: {Target_Posture:>3}', end = '')

        keys = key_check()
        if 'P' in keys:    # 结束
            cv2.destroyAllWindows()
            break
        elif 'T' in keys:    # 切换状态(暂停\继续)
            if paused:
                paused = False
                print('\nStarting!')
                cv2.destroyAllWindows()
                time.sleep(1)
            else:
                paused = True
                print('\nPausing!')
                cv2.destroyAllWindows()
                time.sleep(1)

    print('\nDone!')