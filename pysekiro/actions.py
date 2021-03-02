import threading
import time

import numpy as np

from pysekiro.direct_keys import PressKey, ReleaseKey

# ---*---

# direct keys
W = 0x11
S = 0x1F
A = 0x1E
D = 0x20
# R = 0x13 # 使用道具 | Use Item
# F = 0x21 # 钩绳 | Grappling Hook
J = 0x24
K = 0x25
SPACE = 0x39
LSHIFT = 0x2A
# LCONTROL = 0x1D # 使用义手忍具 | Use Prosthetic Tool

Y = 0x15

# ---*---

def Move_Forward():
    print('\r\t\t\tMove Forward', end='')
    PressKey(W)
    time.sleep(1)
    ReleaseKey(W)

# def Move_Back():
#     print('\r\t\t\tMove Back', end='')
#     PressKey(S)
#     time.sleep(1)
#     ReleaseKey(S)

# def Move_Left():
#     print('\r\t\t\tMove Left', end='')
#     PressKey(A)
#     time.sleep(1)
#     ReleaseKey(A)

# def Move_Right():
#     print('\r\t\t\tMove Right', end='')
#     PressKey(D)
#     time.sleep(1)
#     ReleaseKey(D)

# def Lock_On():
#     print('\r\t\t\tStep Dodge', end='')
#     PressKey(Y)
#     time.sleep(0.1)
#     ReleaseKey(Y)

def Step_Dodge():
    print('\r\t\t\tStep Dodge', end='')
    PressKey(LSHIFT)
    time.sleep(0.1)
    ReleaseKey(LSHIFT)

def Jump():
    print('\r\t\t\tJump', end='')
    PressKey(SPACE)
    time.sleep(0.1)
    ReleaseKey(SPACE)

def Attack():
    print('\r\t\t\tAttack', end='')
    PressKey(J)
    time.sleep(0.1)
    ReleaseKey(J)

def Deflect():
    print('\r\t\t\tDeflect', end='')
    PressKey(K)
    time.sleep(0.08)
    ReleaseKey(K)

# ---*---

# 根据 collect_data.py
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
        act = Move_Forward # 其他
    
    # 暂时只向前移动，其他走位以后再考虑
    move_process = threading.Thread(target=Move_Forward)
    move_process.start()

    act_process = threading.Thread(target=act)
    act_process.start()