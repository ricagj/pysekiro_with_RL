import cv2
import numpy as np

from pysekiro.img_tools.get_vertices import roi

# ---*---

# 获取数值
def get_value(target_img):
    count = 0
    for pixel in target_img:
        if pixel == 0:
            break
        count += 1
    return count

# ---*---

# max 257
def get_Self_HP(img):
    img_roi = roi(img, x=48, x_w=305, y=409, y_h=409+1)    # x, x_w, y, y_h 获取自 get_vertices.py
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离
    g = g[0]
    retval, img_th = cv2.threshold(g, 50, 255, cv2.THRESH_TOZERO)              # 图像阈值处理，像素点的值低于50的设置为0
    retval, img_th = cv2.threshold(img_th, 70, 255, cv2.THRESH_TOZERO_INV)    # 图像阈值处理，像素点的值高于70的设置为0
    img_th = np.reshape(img_th, g.shape)
    Self_HP = get_value(img_th)    # 获取数值
#     print('\n', img_th)
#     print(Self_HP)
    return Self_HP

# max 82
def get_Self_Posture(img):
    img_roi = roi(img, x=401, x_w=483, y=389, y_h=389+1)    # x, x_w, y, y_h 获取自 get_vertices.py
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离
    r = r[0]
    if r[0] == 255:
        retval, img_th = cv2.threshold(r, 200, 255, cv2.THRESH_TOZERO)    # 图像阈值处理，像素点的值低于200的设置为0
        img_th = np.reshape(img_th, r.shape)
        Self_Posture = get_value(img_th)
    elif 155 < r[0] < 170:
        retval, img_th = cv2.threshold(r, 100, 255, cv2.THRESH_TOZERO)    # 图像阈值处理，像素点的值低于100的设置为0
        img_th = np.reshape(img_th, r.shape)
        Self_Posture = get_value(img_th)
    else:
        img_th = None
        Self_Posture = 0
#     print('\n', img_th)
#     print(Self_Posture)
    return Self_Posture

# ---*---

# max 168
def get_Target_HP(img):
    img_roi = roi(img, x=48, x_w=216, y=41, y_h=41+1)    # x, x_w, y, y_h 获取自 get_vertices.py
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离
    g = g[0]
    retval, img_th = cv2.threshold(g, 25, 255, cv2.THRESH_TOZERO)             # 图像阈值处理，像素点的值低于40的设置为0
    retval, img_th = cv2.threshold(img_th, 70, 255, cv2.THRESH_TOZERO_INV)    # 图像阈值处理，像素点的值高于70的设置为0
    img_th = np.reshape(img_th, g.shape)
    Target_HP = get_value(img_th)    # 获取数值
#     print('\n', img_th)
#     print(Target_HP)
    return Target_HP

# max 143
def get_Target_Posture(img):
    img_roi = roi(img, x=401, x_w=544, y=29, y_h=29+1)    # x, x_w, y, y_h 获取自 get_vertices.py
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离
    r = r[0]
    if r[0] == 255:
        retval, img_th = cv2.threshold(r, 160, 255, cv2.THRESH_TOZERO)    # 图像阈值处理，像素点的值低于200的设置为0
        img_th = np.reshape(img_th, r.shape)
        Target_Posture = get_value(img_th)
    elif r[0] > 190:
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

def get_status(img):
    Self_HP, Self_Posture, Target_HP, Target_Posture = get_Self_HP(img), get_Self_Posture(img), get_Target_HP(img), get_Target_Posture(img)
    status_info = f'Self HP: {Self_HP:>3}, Self Posture: {Self_Posture:>3}, Target HP: {Target_HP:>3}, Target Posture: {Target_Posture:>3}. '
    return Self_HP, Self_Posture, Target_HP, Target_Posture, status_info