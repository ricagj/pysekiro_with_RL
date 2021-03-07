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
    img_roi = roi(img, x=29, x_w=182, y=246, y_h=246+1)[0]    # 获取自 get_vertices.py
    retval, img_th = cv2.threshold(img_roi, 60, 255, cv2.THRESH_TOZERO)           # 图像阈值处理，像素点的值低于60的设置为0
    retval, img_th = cv2.threshold(img_th, 80, 255, cv2.cv2.THRESH_TOZERO_INV)    # 图像阈值处理，像素点的值高于80的设置为0
    img_th = np.reshape(img_th, (img_roi.shape))
    Self_HP = get_value(img_th)    # 获取数值
#     print('\n', img_th)
#     print(Self_HP)
    return Self_HP

def get_Self_Posture(img):
    img_roi = roi(img, x=240, x_w=290, y=234, y_h=234+1)[0]    # 获取自 get_vertices.py
    retval, img_th = cv2.threshold(img_roi, 100, 255, cv2.THRESH_TOZERO)    # 图像阈值处理，像素点的值低于100的设置为0
    img_th = np.reshape(img_th, (img_roi.shape))
    
    if int(img_th[0]) - int(img_th[1]) > 15:    # 开启条件
        if img_th[1] in range(100, 125) and img_th[0] in range(145, 165):
            Self_Posture = get_value(img_th)
        elif img_th[1] in range(135, 160) and img_th[0] in range(180, 220):
            Self_Posture = get_value(img_th)
        elif img_th[1] in range(160, 230) and img_th[0] in range(200, 250):
            Self_Posture = get_value(img_th)
        else:
            Self_Posture = 0
    else:
        Self_Posture = 0
#     print('\n', img_th)
#     print(Self_Posture)
    return Self_Posture

# ---*---

def get_Target_HP(img):
    img_roi = roi(img, x=29, x_w=130, y=25, y_h=25+1)[0]    # 获取自 get_vertices.py
    retval, img_th = cv2.threshold(img_roi, 40, 255, cv2.THRESH_TOZERO)           # 图像阈值处理，像素点的值低于40的设置为0
    retval, img_th = cv2.threshold(img_th, 80, 255, cv2.cv2.THRESH_TOZERO_INV)    # 图像阈值处理，像素点的值高于80的设置为0
    img_th = np.reshape(img_th, (img_roi.shape))
    Target_HP = get_value(img_th)    # 获取数值
#     print('\n', img_th)
#     print(Target_HP)
    return Target_HP

def get_Target_Posture(img):
    img_roi = roi(img, x=240, x_w=327, y=17, y_h=17+1)[0]    # 获取自 get_vertices.py
    retval, img_th = cv2.threshold(img_roi, 100, 255, cv2.THRESH_TOZERO)    # 图像阈值处理，像素点的值低于100的设置为0
    img_th = np.reshape(img_th, (img_roi.shape))
    
    if int(img_th[0]) - int(img_th[1]) > 15:    # 开启条件
        if img_th[1] in range(100, 125) and img_th[0] in range(175, 230):
            Target_Posture = get_value(img_th)
        elif img_th[1] in range(150, 190) and img_th[0] in range(220, 250):
            Target_Posture = get_value(img_th)
        elif img_th[1] in range(190, 220) and img_th[0] in range(230, 260):
            Target_Posture = get_value(img_th)
        else:
            Target_Posture = 0
    else:
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