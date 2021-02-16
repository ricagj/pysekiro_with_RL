import threading
import time

import numpy as np

from pysekiro.direct_keys import PressKey, ReleaseKey

# ---*---

W = 0x11
S = 0x1F
A = 0x1E
D = 0x20
R = 0x13 # 使用道具 | Use Item
F = 0x21 # 钩绳 | Grappling Hook
J = 0x24
K = 0x25
SPACE = 0x39
LSHIFT = 0x2A
LCONTROL = 0x1D # 使用义手忍具 | Use Prosthetic Tool

Y = 0x15

# ---*---

def ReleaseAllKey():
    ReleaseKey(J)
    ReleaseKey(K)
    ReleaseKey(LSHIFT)
    ReleaseKey(SPACE)
    ReleaseKey(W)

def Attack():
    print('\t\t\tAttack\tstart')
    PressKey(W)
    PressKey(J)
    time.sleep(0.1)
    ReleaseAllKey()
    print('\t\t\tAttack\t\tstop')

def Deflect():
    print('\t\t\tDeflect\tstart')
    PressKey(W)
    PressKey(K)
    time.sleep(0.08)
    ReleaseAllKey()
    print('\t\t\tDeflect\t\tstop')

def Step_Dodge():
    print('\t\t\tStep Dodge\tstart')
    PressKey(W)
    PressKey(LSHIFT)
    time.sleep(0.1)
    ReleaseAllKey()
    print('\t\t\tStep Dodge\t\tstop')

def Jump():
    print('\t\t\tJump\tstart')
    PressKey(W)
    PressKey(SPACE)
    time.sleep(0.1)
    ReleaseAllKey()
    print('\t\t\tJump\t\tstop')

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