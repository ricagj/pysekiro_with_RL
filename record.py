import os
import time

import cv2

from pysekiro.img_tools.grab_screen import get_screen, get_full_screen
from pysekiro.key_tools.get_keys import key_check

class Record:
    def __init__(self, mode=''):
        self.mode = mode
        if self.mode == 'game':
            self.size = (800, 450)
        else:
            self.size = (1920, 1080)
        self.fps = 20
        
        self.save_dir = 'Video'
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.video = cv2.VideoWriter(
            self.get_save_path(),
            cv2.VideoWriter_fourcc('X','V','I','D'),
            self.fps,
            self.size,
            True
        )

    def get_save_path(self):
        n = 1
        while True:    # 直到找到保存位置并保存就 break
            filename = str(n).zfill(3) + '.avi'
            save_path = os.path.join(self.save_dir, filename)
            if not os.path.exists(save_path):    # 没有重复的文件名就执行保存并退出
                print(save_path)
                return save_path
            n += 1

    def start(self):

        print('Ready!')
        paused = True
        while True:
            keys = key_check()
            last_time = time.time()
            if  paused:
                if 'T' in keys:
                    paused = False
                    time.sleep(1)
                    print('Continuing!')
            else:
                # if 'T' in keys:
                #     paused = True
                #     time.sleep(1)
                #     print('Pausing!')
                
                if self.mode == 'game':
                    screen = get_screen()
                else:
                    screen = get_full_screen()
                self.video.write(screen)
                
                t = 1/self.fps - (time.time() - last_time)
                if t > 0:
                    time.sleep(t)
                print(f'\r loop {time.time() - last_time}', end='')
            
            if 'P' in keys:    # 结束
                self.video.release()
                cv2.destroyAllWindows()
                print('\nDone!')
                break

if __name__ == '__main__':
    mode = input("game or other: ")
    r = Record(mode)
    r.start()