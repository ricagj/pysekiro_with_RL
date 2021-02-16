# -*- coding:utf-8 -*-

import os
import shutil
import time

import cv2
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

from pysekiro.getkeys import key_check
from pysekiro.grabscreen import get_screen

# ---*---

j  = [1,0,0,0,0] # 攻击 | Attack
k  = [0,1,0,0,0] # 弹反 | Deflect
ls = [0,0,1,0,0] # 垫步 | Step Dodge
sp = [0,0,0,1,0] # 跳跃 | Jump
ot = [0,0,0,0,1] # 其他 | Other

def keys_to_output(keys):
    output = [0,0,0,0,0]

    if   'J' in keys:
        output = j
        action = 'Attack'     # 攻击
    elif 'K' in keys:
        output = k
        action = 'Deflect'    # 弹反
    elif 'LSHIFT' in keys:
        output = ls
        action = 'Step Dodge' # 垫步
    elif 'SPACE' in keys:
        output = sp
        action = 'Jump'       # 跳跃
    else:
        output = ot
        action = 'O'      # 其他
    return output, action

# ---*---

# find_max_num() 和 merge_data() 是保存数据时用的代码
# find_max_num() and merge_data() are the codes used when saving data 
def find_max_num(path):
    filenames = os.listdir(path)
    if 'training_data-1.npy' in filenames:
        max_num = max([int(x[14:-4]) for x in filenames if '.npy' in x])
    else:
        max_num = 0
    return max_num

def merge_data(boss):
    path_1 = 'tmp_data'
    max_num_1 = find_max_num(path_1)
    if max_num_1 <= 1:
        print('There is no data to merge.')
        shutil.rmtree(path_1)
        return -1
    
    npy_file = os.path.join(path_1, 'training_data-1.npy')
    data = np.load(npy_file, allow_pickle=True)
    for i in range(2, max_num_1 + 1):
        npy_file = os.path.join(path_1, f'training_data-{i}.npy')
        next_data = np.load(npy_file, allow_pickle=True)
        
        data = np.append(data, next_data, axis=0)
    
    path_2 = os.path.join('The_battle_memory', boss)

    max_num_2 = find_max_num(path_2)
    np.save(os.path.join(path_2, f'training_data-{max_num_2+1}.npy'), data)
    
    shutil.rmtree(path_1)

# ---*---

def collect_data(boss):
    
    starting_value = 1
    training_data = []
    battle_logs = []
    
    path_1 = 'tmp_data'
    if path_1 not in os.listdir():
        os.mkdir(path_1)
    
    save_path = os.path.join(path_1, f'training_data-{starting_value}.npy')
    
    print('Ready!')

    paused = True
    while True:
        
        if not paused:
            last_time = time.time()
            
            screen = get_screen()
            
            # 按键检测
            # key check
            keys = key_check()
            output, action = keys_to_output(keys)
            
            # 数据整合
            # data integration
            training_data.append([screen, output])
            
            # 临时保存数据
            # Save data temporarily
            if len(training_data) == 100:
                np.save(save_path, training_data)
                training_data = []
                starting_value += 1
                save_path = os.path.join(path_1, f'training_data-{starting_value}.npy')
            
            # 记录战斗数据
            # Record combat data
            battle_log = f'\rloop took {round(time.time()-last_time, 3):>5} seconds. action {action:<11}. keys {str(keys):<35}'
            battle_logs.append(battle_log+'\n')
            print(battle_log, end='')
        
        # 再次检测按键
        # key check again
        keys = key_check()
        
        # 按 ‘P’ 结束并保存
        # Press 'P' to stop and save
        if 'P' in keys:
            np.save(save_path, training_data)
            break

        # 按 ‘T’ 暂停或继续
        # Press 'T' to pause or continue
        elif 'T' in keys:
            if paused:
                paused = False
                print('\nStarting!')
                time.sleep(1)
            else:
                paused = True
                print('\nPausing!')
                time.sleep(1)
    
    print('\n\nStop, please wait')
    
    # 合并临时数据
    # Merge temporary data
    merge_data(boss)
    
    # 写入战斗数据到txt
    # Write battle data to TXT
    with open(f'The_battle_memory_of_{boss}.txt', 'a+') as log:
        log.write('\n\n\n\n\n')
        for log_data in battle_logs:
            log.write(log_data)

    print('Done!')

# ---*---

if __name__ == '__main__':
    boss1 = 'Genichiro_Ashina' # 苇名弦一郎
    boss2 = 'Inner_Genichiro'  # 心中的弦一郎
    boss3 = 'Inner_Isshin'     # 心中的一心
    boss4 = 'Isshin,_the_Sword_Saint' # 剑圣 苇名一心
    
    collect_data(boss1)