# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

import win32api as wapi

# virtual keys (见 keys.py.txt文件里的第145行至第254行)
# virtual keys (see lines 145 to 254 in file "keys.py.txt")
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
    
    'T'           : 0x54,
    'P'           : 0x50
}

r"""
2021-02-05
目前只检测以下的键盘按键
	J      攻击
	K      防御
	SPACE  跳跃
	LSHIFT 垫步
	T      暂停\继续 | Pause or continue
	P      结束 | stop
"""
def key_check():
    keys = []
    for key in ['J', 'K', 'SPACE', 'LSHIFT', 'T', 'P']:
        if wapi.GetAsyncKeyState(vk[key]):
            keys.append(key)
    return keys