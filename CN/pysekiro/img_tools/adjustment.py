import cv2

from pysekiro.img_tools.get_status import get_status
from pysekiro.img_tools.get_vertices import roi
from pysekiro.img_tools.grab_screen import get_screen
from pysekiro.key_tools.get_keys import key_check

x   = 250
x_w = 550
y   = 75
y_h = 375

def main():

    paused = True
    print("Ready!")

    while True:
        keys = key_check()
        if paused:
            if 'T' in keys:
                paused = False
                print('Starting!')
        else:

            screen = get_screen()    # 获取屏幕图像

            status_info = get_status(screen)[4]
            print('\r' + status_info, end='')    # 显示状态信息

            cv2.imshow('roi', roi(screen, x, x_w, y, y_h))

            # 校准线
            screen[409:, [48, 49, 304, 305], :] = 255    # 自身生命

            # screen[389, 401:483, :] = 255    # 自身架势
            screen[[384, 385, 392,393], 401:483, :] = 255    # 自身架势
            screen[389:, 401, :] = 255    # 自身架势中线

            screen[:41, [48, 49, 215, 216], :] = 255    # 目标生命

            # screen[29, 401:544, :] = 255    # 目标架势
            screen[[25, 26, 32, 33], 401:544, :] = 255    # 目标架势
            screen[:29, 401, :] = 255    # 目标架势中线

            cv2.imshow('screen', screen)
            cv2.waitKey(1)

            if 'P' in keys:
                cv2.destroyAllWindows()
                break

    print('\nDone!')