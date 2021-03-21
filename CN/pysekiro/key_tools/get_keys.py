# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

import win32api as wapi

# virtual keys
vk = {
    # 'W' : 0x57,
    # 'S' : 0x53,
    # 'A' : 0x41,
    # 'D' : 0x44,
    'LSHIFT' : 0xA0,
    'SPACE'  : 0x20,

    'J' : 0x4A,
    'K' : 0x4B,

    'T' : 0x54,
    'P' : 0x50
}

def key_check():
    keys = []
    for key in ['LSHIFT', 'SPACE', 'J', 'K', 'T', 'P']:    # 'W', 'S', 'A', 'D', 
        if wapi.GetAsyncKeyState(vk[key]):
            keys.append(key)
    return keys