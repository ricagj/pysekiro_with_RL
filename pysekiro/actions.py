import threading
import time

from pysekiro.direct_keys import PressKey, ReleaseKey

# ---*---

# direct keys
dk = {
    'W' : 0x11,
    'S' : 0x1F,
    'A' : 0x1E,
    'D' : 0x20,
    'LSHIFT' : 0x2A,
    'SPACE'  : 0x39,

    # 'Y' : 0x15,

    'J' : 0x24,
    # 'LCONTROL' : 0x1D,
    'K' : 0x25,
    # 'F' : 0x21,
    'R' : 0x13,
}

# ---*---

# def Move_Forward():
#     PressKey(dk['W'])
#     time.sleep(0.1)
#     ReleaseKey(dk['W'])

# def Move_Back():
#     PressKey(dk['S'])
#     time.sleep(0.1)
#     ReleaseKey(dk['S'])

# def Move_Left():
#     PressKey(dk['A'])
#     time.sleep(0.1)
#     ReleaseKey(dk['A'])

# def Move_Right():
#     PressKey(dk['D'])
#     time.sleep(0.1)
#     ReleaseKey(dk['D'])

def Step_Dodge():
    PressKey(dk['LSHIFT'])
    time.sleep(0.1)
    ReleaseKey(dk['LSHIFT'])

def Jump():
    PressKey(dk['SPACE'])
    time.sleep(0.1)
    ReleaseKey(dk['SPACE'])


# def Lock_On():
#     PressKey(dk['Y'])
#     time.sleep(0.1)
#     ReleaseKey(dk['Y'])


def Attack():
    PressKey(dk['J'])
    time.sleep(0.1)
    ReleaseKey(dk['J'])

# def Use_Prosthetic_Tool():
#     PressKey(dk['LCONTROL'])
#     time.sleep(0.1)
#     ReleaseKey(dk['LCONTROL'])

def Deflect():
    PressKey(dk['K'])
    time.sleep(0.1)
    ReleaseKey(dk['K'])

# def Grappling_Hook():
#     PressKey(dk['F'])
#     time.sleep(0.1)
#     ReleaseKey(dk['F'])

# def Use_Item():
#     PressKey(dk['R'])
#     time.sleep(0.1)
#     ReleaseKey(dk['R'])

def NOKEY():
    ReleaseKey(dk['W'])
    ReleaseKey(dk['S'])
    ReleaseKey(dk['A'])
    ReleaseKey(dk['D'])
    ReleaseKey(dk['LSHIFT'])
    ReleaseKey(dk['SPACE'])
    # ReleaseKey(dk['Y'])
    ReleaseKey(dk['J'])
    # ReleaseKey(dk['LCONTROL'])
    ReleaseKey(dk['K'])
    # ReleaseKey(dk['F'])
    ReleaseKey(dk['R'])

# ---*---

# 根据 collect_data.py 中的 get_output()
def act(action):
    
    if   action == 0:
        act = Attack       # 攻击
    elif action == 1:
        act = Deflect      # 弹反
    elif action == 2:
        act = Step_Dodge   # 垫步
    elif action == 3:
        act = Jump         # 跳跃
    elif action == 4:
        act = NOKEY        # 无键

    act_process = threading.Thread(target=act)
    act_process.start()

# elif action == 5:
#     act = Use_Item     # 使用道具    
# elif action == 6:
#     act = Move_Forward # 移动 前
# elif action == 7:
#     act = Move_Back    # 移动 后
# elif action == 8:
#     act = Move_Left    # 移动 左
# elif action == 9:
#     act = Move_Right   # 移动 右