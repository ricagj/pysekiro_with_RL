# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

import win32api as wapi

# virtual keys
vk = {
    'T' : 0x54,
    'P' : 0x50
}

def key_check():
    keys = []
    for key in ['T', 'P']:
        if wapi.GetAsyncKeyState(vk[key]):
            keys.append(key)
    return keys