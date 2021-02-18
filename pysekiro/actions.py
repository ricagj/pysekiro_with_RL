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
    print('\t\t\tMove Forward')
    PressKey(W)
    time.sleep(1)
    ReleaseKey(W)

# def Move_Back():
#     print('\t\t\tMove Back')
#     PressKey(S)
#     time.sleep(1)
#     ReleaseKey(S)

# def Move_Left():
#     print('\t\t\tMove Left')
#     PressKey(A)
#     time.sleep(1)
#     ReleaseKey(A)

# def Move_Right():
#     print('\t\t\tMove Right')
#     PressKey(D)
#     time.sleep(1)
#     ReleaseKey(D)

# def Lock_On():
#     print('\t\t\tStep Dodge')
#     PressKey(Y)
#     time.sleep(0.1)
#     ReleaseKey(Y)

def Step_Dodge():
    print('\t\t\tStep Dodge')
    PressKey(LSHIFT)
    time.sleep(0.1)
    ReleaseKey(LSHIFT)

def Jump():
    print('\t\t\tJump')
    PressKey(SPACE)
    time.sleep(0.1)
    ReleaseKey(SPACE)

def Attack():
    print('\t\t\tAttack')
    PressKey(J)
    time.sleep(0.1)
    ReleaseKey(J)

def Deflect():
    print('\t\t\tDeflect')
    PressKey(K)
    time.sleep(0.08)
    ReleaseKey(K)

# ---*---

def act(values):

    action = np.argmax(values)
    
    if   action == 0:
        act = Attack     # 攻击
    elif action == 1:
        act = Deflect    # 弹反
    elif action == 2:
        act = Step_Dodge # 垫步
    elif action == 3:
        act = Jump       # 跳跃
    elif action == 4:
        act = ReleaseAllKey # 其他
    
    # 暂时只向前移动，其他走位以后再考虑
    move_process = threading.Thread(target=Move_Forward)
    move_process.start()

    act_process = threading.Thread(target=act)
    act_process.start()

    return action