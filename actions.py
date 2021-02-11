import threading
import time

from directkeys import PressKey, ReleaseKey

# ---*---

# direct keys (见 keys.py.txt文件里的第34行至第143行)
# direct keys (see lines 34 to 143 in file "keys.py.txt")
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
    print('Attack')
    PressKey(W)
    PressKey(J)
    time.sleep(0.07)
    ReleaseAllKey()

def Deflect():
    print('Deflect')
    PressKey(W)
    PressKey(K)
    time.sleep(0.07)
    ReleaseAllKey()

def Step_Dodge():
    print('Step Dodge')
    PressKey(W)
    PressKey(LSHIFT)
    time.sleep(0.07)
    ReleaseAllKey()

def Jump():
    print('Jump')
    PressKey(W)
    PressKey(SPACE)
    time.sleep(0.07)
    ReleaseAllKey()

# ---*---

def action_choice(choice):

    if   choice == 0:
        action = Attack     # 攻击
    elif choice == 1:
        action = Deflect    # 弹反
    elif choice == 2:
        action = Step_Dodge # 垫步
    elif choice == 3:
        action = Jump       # 跳跃
    elif choice == 4:
        action = ReleaseAllKey      # 其他
    
    action_process = threading.Thread(target=action)
    action_process.start()