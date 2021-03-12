# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

import win32api as wapi

# virtual keys
vk = {
    'W' : 0x57,
    'S' : 0x53,
    'A' : 0x41,
    'D' : 0x44,
    'LSHIFT' : 0xA0,
    'SPACE'  : 0x20,

    'J' : 0x4A,
    # 'LCONTROL' : 0xA2,
    'K' : 0x4B,
    #'F' : 0x46,
    'R' : 0x52,

    'T' : 0x54,
    'P' : 0x50
}

keyList = ['\b']

def key_check():
    keys = []
    for key in ['W', 'S', 'A', 'D', 'LSHIFT', 'SPACE', 'J', 'K', 'R', 'T', 'P']:
        if wapi.GetAsyncKeyState(vk[key]):
            keys.append(key)
    return keys