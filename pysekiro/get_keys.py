# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

import win32api as wapi

vk = {
    'W' : 0x57,
    'S' : 0x53,
    'A' : 0x41,
    'D' : 0x44,
    'R' : 0x52,
    'F' : 0x46,
    'J' : 0x4A,
    'K' : 0x4B,
    'SPACE'    : 0x20,
    'LSHIFT'   : 0xA0,
    'LCONTROL' : 0xA2, 
    'T' : 0x54,
    'P' : 0x50
}

keyList = ['\b']

def key_check():
    keys = []
    for key in ['W', 'S', 'A', 'D', 'J', 'K', 'SPACE', 'LSHIFT', 'T', 'P']:
        if wapi.GetAsyncKeyState(vk[key]):
            keys.append(key)
    return keys