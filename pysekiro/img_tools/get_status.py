import cv2
import numpy as np

from pysekiro.img_tools.get_vertices import roi

# ---*---

# 获取数值
def get_value(target_img):
    count = 0
    for i in range(len(target_img)-1):
        cur_pixel = int(target_img[i])
        if cur_pixel == 0:
            break
        count += 1
    return count

# ---*---

def get_Self_HP(img):
    img_roi = roi(img, x=77, x_w=489, y=656, y_h=656+1)    # x, x_w, y, y_h 获取自 get_vertices.py
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离
    g = g[0]
    retval, img_th = cv2.threshold(g, 50, 255, cv2.THRESH_TOZERO)             # 图像阈值处理，像素点的值低于50的设置为0
    retval, img_th = cv2.threshold(img_th, 70, 255, cv2.THRESH_TOZERO_INV)    # 图像阈值处理，像素点的值高于70的设置为0
    img_th = np.reshape(img_th, g.shape)
    Self_HP = get_value(img_th)    # 获取数值
#     print('\n', img_th)
#     print(Self_HP)
    return Self_HP

def get_Self_Posture(img):
    img_roi = roi(img, x=641, x_w=775, y=625, y_h=625+1)    # x, x_w, y, y_h 获取自 get_vertices.py
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离
    r = r[0]
    if 169 < r[0] < 173:    # 开启条件1
        retval, img_th = cv2.threshold(r, 120, 255, cv2.THRESH_TOZERO)    # 图像阈值处理，像素点的值低于120的设置为0
        img_th = np.reshape(img_th, r.shape)
        Self_Posture = get_value(img_th)
    elif r[0] > 253:    # 开启条件2
        retval, img_th = cv2.threshold(r, 220, 255, cv2.THRESH_TOZERO)    # 图像阈值处理，像素点的值低于220的设置为0
        img_th = np.reshape(img_th, r.shape)
        Self_Posture = get_value(img_th)
    else:
        img_th = None
        Self_Posture = 0
    # print('\n', img_th)
    # print(Self_Posture)
    return Self_Posture

# ---*---

def get_Target_HP(img):
    img_roi = roi(img, x=77, x_w=349, y=67, y_h=67+1)    # x, x_w, y, y_h 获取自 get_vertices.py
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离
    g = g[0]
    retval, img_th = cv2.threshold(g, 60, 255, cv2.THRESH_TOZERO_INV)           # 图像阈值处理，像素点的值高于60的设置为0
    img_th = np.reshape(img_th, g.shape)
    Target_HP = get_value(img_th)    # 获取数值
#     print('\n', img_th)
#     print(Target_HP)
    return Target_HP

def get_Target_Posture(img):
    img_roi = roi(img, x=641, x_w=874, y=46, y_h=46+1)    # x, x_w, y, y_h 获取自 get_vertices.py
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离
    r = r[0]
    if r[0] > 200:    # 开启条件1
        retval, img_th = cv2.threshold(r, 130, 255, cv2.THRESH_TOZERO)    # 图像阈值处理，像素点的值低于130的设置为0
        img_th = np.reshape(img_th, r.shape)
        Target_Posture = get_value(img_th)
    else:
        img_th = None
        Target_Posture = 0
#     print('\n', img_th)
#     print(Target_Posture)
    return Target_Posture

# ---*---

def get_status(img, show=False):
    Self_HP, Self_Posture, Target_HP, Target_Posture = get_Self_HP(img), get_Self_Posture(img), get_Target_HP(img), get_Target_Posture(img)
    if show:
        print(f'\rSelf HP: {Self_HP:>3}, Self Posture: {Self_Posture:>3}, Target HP: {Target_HP:>3}, Target Posture: {Target_Posture:>3}. ', end='')
    return Self_HP, Self_Posture, Target_HP, Target_Posture