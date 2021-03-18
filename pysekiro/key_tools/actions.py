# import threading
import time

from pysekiro.key_tools.direct_keys import PressKey, ReleaseKey

# ---*---

# direct keys
dk = {
    'W' : 0x11,
    'S' : 0x1F,
    'A' : 0x1E,
    'D' : 0x20,
    'LSHIFT' : 0x2A,
    'SPACE'  : 0x39,

    'Y' : 0x15,

    'J' : 0x24,
    'K' : 0x25,
}

# 设置动作本身执行所需的时间
# 注，为了数据的稳定性，延时时间要统一
delay = 0.01

# ---*---

# def Move_Forward():
#     PressKey(dk['W'])
#     time.sleep(delay)
#     ReleaseKey(dk['W'])

# def Move_Back():
#     PressKey(dk['S'])
#     time.sleep(delay)
#     ReleaseKey(dk['S'])

# def Move_Left():
#     PressKey(dk['A'])
#     time.sleep(delay)
#     ReleaseKey(dk['A'])

# def Move_Right():
#     PressKey(dk['D'])
#     time.sleep(delay)
#     ReleaseKey(dk['D'])

def Step_Dodge():    # 0.605
    PressKey(dk['LSHIFT'])
    time.sleep(delay)
    ReleaseKey(dk['LSHIFT'])

def Jump():    # 1.101
    PressKey(dk['SPACE'])
    time.sleep(delay)
    ReleaseKey(dk['SPACE'])


def Lock_On():
    PressKey(dk['Y'])
    time.sleep(delay)
    ReleaseKey(dk['Y'])


def Attack():    # 0.640
    PressKey(dk['J'])
    time.sleep(delay)
    ReleaseKey(dk['J'])

def Deflect():    # 0.199
    PressKey(dk['K'])
    time.sleep(delay)
    ReleaseKey(dk['K'])

def NOKEY():
    time.sleep(delay)
    ReleaseKey(dk['W'])
    ReleaseKey(dk['S'])
    ReleaseKey(dk['A'])
    ReleaseKey(dk['D'])
    ReleaseKey(dk['LSHIFT'])
    ReleaseKey(dk['SPACE'])
    ReleaseKey(dk['Y'])
    ReleaseKey(dk['J'])
    ReleaseKey(dk['K'])

# ---*---

def act(action=4, WS=2, AD=2):
    
    if   action == 0:
        act = Attack       # 攻击
    elif action == 1:
        act = Deflect      # 弹反
    elif action == 2:
        act = Step_Dodge   # 垫步
    elif action == 3:
        act = Jump         # 跳跃
    else:
        act = NOKEY        # 无键, 无动作
    
    PressKey(dk['W'])
    act()
    ReleaseKey(dk['W'])

    # if   WS == 0:
    #     ws = Move_Forward # 移动 前
    # elif WS == 1:
    #     ws = Move_Back    # 移动 后
    # else:
    #     ws = NOKEY        # 无键, 无动作
    # ws_process = threading.Thread(target=ws)
    # ws_process.start()

    # if   AD == 0:
    #     ad = Move_Left    # 移动 左
    # elif AD == 1:
    #     ad = Move_Right   # 移动 右
    # else:
    #     ad = NOKEY        # 无键, 无动作
    # ad_process = threading.Thread(target=ad)
    # ad_process.start()