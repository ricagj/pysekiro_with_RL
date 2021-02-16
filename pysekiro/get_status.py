import os

import cv2
import numpy as np

from get_vertices.py import roi

# ---*---

def get_HP(target_img):
    count = 0
    
    if target_img[0] == 0 or target_img[1] == 0:
        return count
    
    for i in range(len(target_img)-1):
        cur_pixel = int(target_img[i])
        next_pixel = int(target_img[i+1])
        if abs(cur_pixel - next_pixel) > 20 or cur_pixel < 40 or cur_pixel > 80:
            break
        count += 1
    return count

def get_Posture(target_img):
    count = 0
    
    if target_img[0] == 0 or target_img[1] == 0:
        return count
    
    for i in range(len(target_img)-1):
        cur_pixel = int(target_img[i])
        next_pixel = int(target_img[i+1])
        if abs(cur_pixel - next_pixel) > 20 or cur_pixel < 100:
            break
        count += 1
    return count

# ---*---

def get_Sekiro_HP(img):
    img_roi = roi(img, x=29, x_w=182, y=244, y_h=246)[0]
    Sekiro_HP = get_HP(img_roi)
#     print('\n\n', img_roi, Sekiro_HP)
    return Sekiro_HP

def get_Sekiro_Posture(img):
    img_roi = roi(img, x=241, x_w=290, y=233, y_h=235)[0]
    Sekiro_Posture = get_Posture(img_roi)
#     print('\n\n', img_roi, Sekiro_Posture)
    return Sekiro_Posture

def get_Boss_HP(img):
    img_roi = roi(img, x=29, x_w=129, y=24, y_h=26)[0]
    Boss_HP = get_HP(img_roi)
#     print('\n\n', img_roi, Boss_HP)
    return Boss_HP

def get_Boss_Posture(img):
    img_roi = roi(img, x=241, x_w=326, y=16, y_h=18)[0]
    Boss_Posture = get_Posture(img_roi)
#     print('\n\n', img_roi, Boss_Posture)
    return Boss_Posture

# ---*---

def get_status(img):
    return [get_Sekiro_HP(img), get_Sekiro_Posture(img), get_Boss_HP(img), get_Boss_Posture(img)]

# ---*---

def main():

    print('Press "q" to quit. ') # 按q键离开。

    boss = 'Genichiro_Ashina' # 苇名弦一郎
    data = np.load(os.path.join('The_battle_memory', boss, f'training_data-{105}.npy'), allow_pickle=True)

    # data = data[1500:1700]
    Remaining = len(data)

    for screen, action_value in data:

        if   action_value == [1,0,0,0,0]:
            motion = 'Attack'     # 攻击
        elif action_value == [0,1,0,0,0]:
            motion = 'Deflect'    # 弹反
        elif action_value == [0,0,1,0,0]:
            motion = 'Step Dodge' # 垫步
        elif action_value == [0,0,0,1,0]:
            motion = 'Jump'       # 跳跃
        elif action_value == [0,0,0,0,1]:
            motion = 'O'      # 其他

        cv2.imshow('screen', screen)

        Sekiro_HP, Sekiro_Posture, Boss_HP, Boss_Posture = get_status(screen)

        Remaining -= 1
        print(f'\r Remaining: {Remaining:<6}, motion:{motion:<11}, Sekiro_HP: {Sekiro_HP:>4}, Sekiro_Posture: {Sekiro_Posture:>4}, Boss_HP:{Boss_HP:>4}, Boss_Posture: {Boss_Posture:>4}', end='')
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    else:
        cv2.destroyAllWindows()

# ---*---

if __name__ == '__main__':
    main()