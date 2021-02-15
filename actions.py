import threading
import time

import numpy as np

from directkeys import PressKey, ReleaseKey

# ---*---

W = 0x11 # 移动 前 | Move Forward
S = 0x1F # 移动 后 | Move Back
A = 0x1E # 移动 左 | Move Left
D = 0x20 # 移动 右 | Move Right
R = 0x13 # 使用道具 | Use Item
F = 0x21 # 钩绳 | Grappling Hook
J = 0x24 # 攻击 | Attack
K = 0x25 # 防御 | Deflect, (Hold) Guard
SPACE = 0x39    # 跳跃 | Jump
LSHIFT = 0x2A   # 垫步、（长按）冲刺 | Step Dodge, (hold) Sprint
LCONTROL = 0x1D # 使用义手忍具 | Use Prosthetic Tool

# ---*---

def ReleaseAllKey():
    ReleaseKey(J)
    ReleaseKey(K)
    ReleaseKey(LSHIFT)
    ReleaseKey(SPACE)
    ReleaseKey(W)

def Attack():
    # print('\t\t\tAttack\t\t\tstart')
    PressKey(W)
    PressKey(J)
    time.sleep(0.08)
    ReleaseAllKey()
    # print('\t\t\tAttack\t\t\tstop')

def Deflect():
    # print('\t\t\tDeflect\t\t\tstart')
    PressKey(W)
    PressKey(K)
    time.sleep(0.08)
    ReleaseAllKey()
    # print('\t\t\tDeflect\t\t\tstop')

def Step_Dodge():
    # print('\t\t\tStep Dodge\t\t\tstart')
    PressKey(W)
    PressKey(LSHIFT)
    time.sleep(0.08)
    ReleaseAllKey()
    # print('\t\t\tStep Dodge\t\t\tstop')

def Jump():
    # print('\t\t\tJump\t\t\tstart')
    PressKey(W)
    PressKey(SPACE)
    time.sleep(0.08)
    ReleaseAllKey()
    # print('\t\t\tJump\t\t\tstop')

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
    
    act_process = threading.Thread(target=act)
    act_process.start()

    return action