import cv2

from pysekiro.get_keys import key_check
from pysekiro.grab_screen import get_screen

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

            screen[246:,[29, 182]] = 255
            screen[[233, 235],240:290] = 255
            screen[:25,[29, 130]] = 255
            screen[[16, 18],240:327] = 255
            cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
            cv2.imshow('screen', screen)
            cv2.waitKey(1)

            if 'P' in keys:    # 结束
                cv2.destroyAllWindows()
                break

    print('\nDone!')