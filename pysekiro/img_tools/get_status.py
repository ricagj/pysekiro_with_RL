import cv2
import numpy as np

from pysekiro.img_tools.get_vertices import roi

# ---*---

# 获取数值
def get_value(target_img):
    count = 0
    for pixel in target_img[0]:
        if pixel == 0:
            break
        count += 1
    return count

# ---*---

# 获取自身生命 (max 257)
def get_Self_HP(img):
    img_roi = roi(img, x=48, x_w=305, y=409, y_h=409+1)    # x, x_w, y, y_h 获取自 get_vertices.py
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离
    retval, img_th = cv2.threshold(g, 50, 255, cv2.THRESH_TOZERO)              # 图像阈值处理，像素点的值低于50的设置为0
    retval, img_th = cv2.threshold(img_th, 70, 255, cv2.THRESH_TOZERO_INV)    # 图像阈值处理，像素点的值高于70的设置为0
    Self_HP = get_value(img_th)    # 获取数值
    return Self_HP

# ---*---

# up to data
# 获取自身架势 (max 82)
def get_Self_Posture(img):
    img_roi = roi(img, x=401, x_w=483, y=389, y_h=389+1)    # x, x_w, y, y_h 获取自 get_vertices.py
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离
    white_line = r[0][0]
    if 155 < white_line < 170 or white_line > 250:
        canny = cv2.Canny(cv2.GaussianBlur(r,(3,3),0), 0, 100)
        Self_Posture =  np.argmax(canny)
    else:
        Self_Posture = 0
    return Self_Posture

# outdata
# 获取自身架势 (max 82)
def o_get_Self_Posture(img):
    img_roi = roi(img, x=401, x_w=483, y=389, y_h=389+1)    # x, x_w, y, y_h 获取自 get_vertices.py
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离
    if r[0][0] > 250:
        retval, img_th = cv2.threshold(r, 200, 255, cv2.THRESH_TOZERO)    # 图像阈值处理，像素点的值低于200的设置为0
        Self_Posture = get_value(img_th)
    elif 155 < r[0][0] < 170:
        retval, img_th = cv2.threshold(r, 100, 255, cv2.THRESH_TOZERO)    # 图像阈值处理，像素点的值低于100的设置为0
        # retval, img_th = cv2.threshold(img_th, 170, 255, cv2.THRESH_TOZERO_INV)    # 图像阈值处理，像素点的值高于170的设置为0
        Self_Posture = get_value(img_th)
    else:
        img_th = None
        Self_Posture = 0
    # print('\n', 'Self_Posture', img_th)
#     print(Self_Posture)
    return Self_Posture

# ---*---

# 获取目标生命 (max 168)
def get_Target_HP(img):
    img_roi = roi(img, x=48, x_w=216, y=41, y_h=41+1)    # x, x_w, y, y_h 获取自 get_vertices.py
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离
    retval, img_th = cv2.threshold(g, 25, 255, cv2.THRESH_TOZERO)             # 图像阈值处理，像素点的值低于40的设置为0
    retval, img_th = cv2.threshold(img_th, 70, 255, cv2.THRESH_TOZERO_INV)    # 图像阈值处理，像素点的值高于70的设置为0
    Target_HP = get_value(img_th)    # 获取数值
    # print('\n', 'Target_HP', img_th)
#     print(Target_HP)
    return Target_HP

# ---*---

# up to data
# 获取目标架势 (max 143)
def get_Target_Posture(img):
    img_roi = roi(img, x=401, x_w=544, y=29, y_h=29+1)    # x, x_w, y, y_h 获取自 get_vertices.py
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离
    white_line = r[0][0]
    if white_line > 190:
        canny = cv2.Canny(cv2.GaussianBlur(r,(3,3),0), 0, 100)
        Target_Posture =  np.argmax(canny)
    else:
        Target_Posture = 0
    return Target_Posture

# outdata
# 获取目标架势 (max 143)
def o_get_Target_Posture(img):
    img_roi = roi(img, x=401, x_w=544, y=29, y_h=29+1)    # x, x_w, y, y_h 获取自 get_vertices.py
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离
    if r[0][0] > 250:
        retval, img_th = cv2.threshold(r, 140, 255, cv2.THRESH_TOZERO)    # 图像阈值处理，像素点的值低于140的设置为0
        Target_Posture = get_value(img_th)
    elif r[0][0] > 190:
        retval, img_th = cv2.threshold(r, 120, 255, cv2.THRESH_TOZERO)    # 图像阈值处理，像素点的值低于120的设置为0
        Target_Posture = get_value(img_th)
    else:
        img_th = None
        Target_Posture = 0
    # print('\n', 'Target_Posture', img_th)
#     print(Target_Posture)
    return Target_Posture

# ---*---

def get_status(img):
    Self_HP, Self_Posture, Target_HP, Target_Posture = get_Self_HP(img), get_Self_Posture(img), get_Target_HP(img), get_Target_Posture(img)
    status_info = f'Self HP: {Self_HP:>3}, Self Posture: {Self_Posture:>3}, Target HP: {Target_HP:>3}, Target Posture: {Target_Posture:>3} . '
    return Self_HP, Self_Posture, Target_HP, Target_Posture, status_info

def o_get_status(img):
    o_Self_Posture, o_Target_Posture = o_get_Self_Posture(img), o_get_Target_Posture(img)
    return o_Self_Posture, o_Target_Posture