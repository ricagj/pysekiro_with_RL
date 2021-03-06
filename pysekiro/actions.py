import threading
import time

import numpy as np

from pysekiro.direct_keys import PressKey, ReleaseKey

# ---*---

# direct keys
dk = {
    'W' : 0x11,
    'S' : 0x1F,
    'A' : 0x1E,
    'D' : 0x20,
    #'R' : 0x13, # 使用道具 | Use Item
    #'F' : 0x21, # 钩绳 | Grappling Hook
    'J' : 0x24,
    'K' : 0x25,
    'SPACE'    : 0x39,
    'LSHIFT'   : 0x2A,
    #'LCONTROL' : 0x1D, # 使用义手忍具 | Use Prosthetic Tool
    # 'Y' : 0x15,
}

# ---*---

def Move_Forward():
    # print('\r\t\t\tMove Forward', end='')
    PressKey(dk['W'])
    time.sleep(1)
    ReleaseKey(dk['W'])

# def Move_Back():
#     PressKey(dk['S'])
#     time.sleep(1)
#     ReleaseKey(dk['S'])

# def Move_Left():
#     PressKey(dk['A'])
#     time.sleep(1)
#     ReleaseKey(dk['A'])

# def Move_Right():
#     PressKey(dk['D'])
#     time.sleep(1)
#     ReleaseKey(dk['D'])

# def Lock_On():
#     PressKey(dk['Y'])
#     time.sleep(0.1)
#     ReleaseKey(dk['Y'])

def Step_Dodge():
    PressKey(dk['LSHIFT'])
    time.sleep(0.1)
    ReleaseKey(dk['LSHIFT'])

def Jump():
    PressKey(dk['SPACE'])
    time.sleep(0.1)
    ReleaseKey(dk['SPACE'])

def Attack():
    PressKey(dk['J'])
    time.sleep(0.1)
    ReleaseKey(dk['J'])

def Deflect():
    PressKey(dk['K'])
    time.sleep(0.1)
    ReleaseKey(dk['K'])

# ---*---

# 根据 collect_data.py 中的 get_output()
def act(values):
    
    if   values == 0:
        act = Attack     # 攻击
    elif values == 1:
        act = Deflect    # 弹反
    elif values == 2:
        act = Step_Dodge # 垫步
    elif values == 3:
        act = Jump       # 跳跃
    elif values == 4:
        act = Move_Forward # 其他（暂时用 移动 前 替代）

    act_process = threading.Thread(target=act)
    act_process.start()