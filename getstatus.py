import os

import cv2
import numpy as np

from getvertices import roi

# ---*---

def get_HP_capacity(target_img):
    count = 0
    
    if target_img[0] == 0:
        return count
    
    for i in range(len(target_img)-1):
        cur_pixel = int(target_img[i])
        next_pixel = int(target_img[i+1])
        if abs(cur_pixel - next_pixel) > 20 or cur_pixel < 40 or cur_pixel > 80:
            break
        count += 1
    return count

def get_Posture_capacity(target_img):
    count = 0
    
    if target_img[0] == 0:
        return count
    
    for i in range(len(target_img)-1):
        cur_pixel = int(target_img[i])
        next_pixel = int(target_img[i+1])
        if abs(cur_pixel - next_pixel) > 20 or cur_pixel < 100:
            break
        count += 1
    return count

# ---*---

def get_Sekiro_HP_Capacity(img):
    Sekiro_HP = roi(img, x=29, x_w=182, y=244, y_h=246)[0]
    Sekiro_HP_Capacity = get_HP_capacity(Sekiro_HP)
#     print('\n\n', Sekiro_HP, Sekiro_HP_Capacity)
    return Sekiro_HP_Capacity

def get_Sekiro_Posture_Capacity(img):
    Sekiro_Posture = roi(img, x=241, x_w=290, y=233, y_h=235)[0]
    Sekiro_Posture_Capacity = get_Posture_capacity(Sekiro_Posture)
#     print('\n\n', Sekiro_Posture, Sekiro_Posture_Capacity)
    return Sekiro_Posture_Capacity

def get_Boss_HP_Capacity(img):
    Boss_HP = roi(img, x=29, x_w=129, y=24, y_h=26)[0]
    Boss_HP_Capacity = get_HP_capacity(Boss_HP)
#     print('\n\n', Boss_HP, Boss_HP_Capacity)
    return Boss_HP_Capacity

def get_Boss_Posture_Capacity(img):
    Boss_Posture = roi(img, x=241, x_w=326, y=16, y_h=18)[0]
    Boss_Posture_Capacity = get_Posture_capacity(Boss_Posture)
#     print('\n\n', Boss_Posture, Boss_Posture_Capacity)
    return Boss_Posture_Capacity

# ---*---

def get_status(img):
    return get_Sekiro_HP_Capacity(img), get_Sekiro_Posture_Capacity(img), get_Boss_HP_Capacity(img), get_Boss_Posture_Capacity(img)

# ---*---

def main():

    print('Press "q" to quit. ') # 按q键离开。

    boss = 'Genichiro_Ashina' # 苇名弦一郎
    data = np.load(os.path.join('The_battle_memory', boss, f'training_data-{105}.npy'), allow_pickle=True)

    # data = data[1500:1700]
    Remaining = len(data)

    for img, cmd in data:

        if   cmd == [1,0,0,0,0]:
            motion = 'Attack'     # 攻击
        elif cmd == [0,1,0,0,0]:
            motion = 'Deflect'    # 弹反
        elif cmd == [0,0,1,0,0]:
            motion = 'Step Dodge' # 垫步
        elif cmd == [0,0,0,1,0]:
            motion = 'Jump'       # 跳跃
        elif cmd == [0,0,0,0,1]:
            motion = 'O'      # 其他

        cv2.imshow('img', img)

        Sekiro_HP, Sekiro_Posture, Boss_HP, Boss_Posture = get_status(img)

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