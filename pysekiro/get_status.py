import cv2
import numpy as np

from pysekiro.get_vertices import roi

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
    img_roi = roi(img, x=29, x_w=182, y=246, y_h=246+1)    # x, x_w, y, y_h 获取自 get_vertices.py
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离
    r = r[0]
    retval, img_th = cv2.threshold(r, 100, 255, cv2.THRESH_TOZERO)           # 图像阈值处理，像素点的值低于100的设置为0
    img_th = np.reshape(img_th, r.shape)
    Self_HP = get_value(img_th)    # 获取数值
    # print('\n', img_th)
    # print(Self_HP)
    return Self_HP

def get_Self_Posture(img):
    img_roi = roi(img, x=240, x_w=290, y=234, y_h=234+1)    # x, x_w, y, y_h 获取自 get_vertices.py
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离
    r = r[0]
    if 150 < r[0] < 165:    # 开启条件1
        retval, img_th = cv2.threshold(r, 130, 255, cv2.THRESH_TOZERO)    # 图像阈值处理，像素点的值低于130的设置为0
        img_th = np.reshape(img_th, r.shape)
        Self_Posture = get_value(img_th)
    elif r[0] > 200:    # 开启条件2
        retval, img_th = cv2.threshold(r, 200, 255, cv2.THRESH_TOZERO)    # 图像阈值处理，像素点的值低于200的设置为0
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
    img_roi = roi(img, x=29, x_w=130, y=25, y_h=25+1)    # x, x_w, y, y_h 获取自 get_vertices.py
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离
    r = r[0]
    retval, img_th = cv2.threshold(r, 80, 255, cv2.THRESH_TOZERO)           # 图像阈值处理，像素点的值低于80的设置为0
    img_th = np.reshape(img_th, r.shape)
    Target_HP = get_value(img_th)    # 获取数值
#     print('\n', img_th)
#     print(Target_HP)
    return Target_HP

def get_Target_Posture(img):
    img_roi = roi(img, x=240, x_w=327, y=17, y_h=17+1)    # x, x_w, y, y_h 获取自 get_vertices.py
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离
    r = r[0]
    if r[0] > 180:    # 开启条件1
        retval, img_th = cv2.threshold(r, 140, 255, cv2.THRESH_TOZERO)    # 图像阈值处理，像素点的值低于140的设置为0
        img_th = np.reshape(img_th, r.shape)
        Target_Posture = get_value(img_th)
    else:
        img_th = None
        Target_Posture = 0
    # print('\n', img_th)
    # print(Target_Posture)
    return Target_Posture

# ---*---

def get_status(img, show=False):
    Self_HP, Self_Posture, Target_HP, Target_Posture = get_Self_HP(img), get_Self_Posture(img), get_Target_HP(img), get_Target_Posture(img)
    if show:
        print(f'\rSelf HP: {Self_HP:>3}, Self Posture: {Self_Posture:>3}, Target HP: {Target_HP:>3}, Target Posture: {Target_Posture:>3}. ', end='')
    return Self_HP, Self_Posture, Target_HP, Target_Posture