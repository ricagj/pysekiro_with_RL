import cv2
import numpy as np

from pysekiro.img_tools.get_vertices import roi

# 存在一些读取数值不准确的问题

# ---*---

# 获取自身生命
def get_Self_HP(img):
    img_roi = roi(img, x=48, x_w=305, y=409, y_h=409+1)    # x, x_w, y, y_h 获取自 get_vertices.py

    b, g ,r =cv2.split(img_roi)    # 颜色通道分离

    retval, img_th = cv2.threshold(g, 50, 255, cv2.THRESH_TOZERO)              # 图像阈值处理，像素点的值低于50的设置为0
    retval, img_th = cv2.threshold(img_th, 70, 255, cv2.THRESH_TOZERO_INV)    # 图像阈值处理，像素点的值高于70的设置为0

    target_img = img_th[0]
    if 0 in target_img:
        Self_HP = np.argmin(target_img)
    else:
        Self_HP = len(target_img)

    return Self_HP

# ---*---

# 获取自身架势
def get_Self_Posture(img):
    img_roi = roi(img, x=401, x_w=490, y=389, y_h=389+1)    # x, x_w, y, y_h 获取自 get_vertices.py
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离

    white_line = r[0][0]
    if 155 < white_line < 170 or white_line > 250:
        canny = cv2.Canny(cv2.GaussianBlur(r,(3,3),0), 0, 100)    # Canny边缘检测
        Self_Posture =  np.argmax(canny)
    else:
        Self_Posture = 0

    if white_line > 250 and Self_Posture < 10:    # 防止满架势却读取为 0
        Self_Posture == len(canny)

    return Self_Posture

# ---*---

# 获取目标生命
def get_Target_HP(img):
    img_roi = roi(img, x=48, x_w=216, y=41, y_h=41+1)    # x, x_w, y, y_h 获取自 get_vertices.py

    b, g ,r =cv2.split(img_roi)    # 颜色通道分离

    retval, img_th = cv2.threshold(g, 25, 255, cv2.THRESH_TOZERO)             # 图像阈值处理，像素点的值低于25的设置为0
    retval, img_th = cv2.threshold(img_th, 70, 255, cv2.THRESH_TOZERO_INV)    # 图像阈值处理，像素点的值高于70的设置为0

    target_img = img_th[0]
    if 0 in target_img:
        Target_HP = np.argmin(target_img)
    else:
        Target_HP = len(target_img)
    
    return Target_HP

# ---*---

# 获取目标架势
def get_Target_Posture(img):
    img_roi = roi(img, x=401, x_w=553, y=29, y_h=29+1)    # x, x_w, y, y_h 获取自 get_vertices.py
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离

    white_line = r[0][0]
    if white_line > 190:
        canny = cv2.Canny(cv2.GaussianBlur(r,(3,3),0), 0, 100)    # Canny边缘检测
        Target_Posture =  np.argmax(canny)
    else:
        Target_Posture = 0

    if white_line > 250 and Target_Posture < 10:    # 防止满架势却读取为 0
        Target_Posture == len(canny)

    return Target_Posture

# ---*---

def get_status(img):
    Self_HP, Self_Posture, Target_HP, Target_Posture = get_Self_HP(img), get_Self_Posture(img), get_Target_HP(img), get_Target_Posture(img)
    status_info = f'Self HP: {Self_HP:>3}, Self Posture: {Self_Posture:>3}, Target HP: {Target_HP:>3}, Target Posture: {Target_Posture:>3} . '
    return Self_HP, Self_Posture, Target_HP, Target_Posture, status_info