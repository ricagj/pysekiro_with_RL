import cv2

from pysekiro.img_tools.get_status import get_status
from pysekiro.img_tools.grab_screen import get_screen
from pysekiro.key_tools.get_keys import key_check

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

            status_info = get_status(screen)[4]    # 显示状态信息
            print('\r' + status_info, end='')

            # 校准线
            screen[653:,[77, 78, 487, 488], :] = 255    # 自身生命

            screen[[620, 621, 630, 631],641:768, :] = 255    # 自身架势
            screen[620:,[641, 642], :] = 255    # 自身架势中线

            screen[:72,[77, 78, 346, 347], :] = 255    # 目标生命

            screen[[41, 42, 50, 51],641:871, :] = 255    # 目标架势
            screen[:51,[641, 642], :] = 255    # 目标架势中线

            cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
            cv2.imshow('screen', screen)
            cv2.waitKey(1)

            if 'P' in keys:    # 结束
                cv2.destroyAllWindows()
                break

    print('\nDone!')