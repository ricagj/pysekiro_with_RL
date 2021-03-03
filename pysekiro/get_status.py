import cv2
import numpy as np

from pysekiro.get_vertices import roi

# ---*---

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
    img_roi = roi(img, x=29, x_w=182, y=246, y_h=246+1)[0]
    retval, img_th = cv2.threshold(img_roi, 60, 255, cv2.THRESH_TOZERO)
    retval, img_th = cv2.threshold(img_th, 80, 255, cv2.cv2.THRESH_TOZERO_INV)
    img_th = np.reshape(img_th, (img_roi.shape))
    Self_HP = get_value(img_th)
#     print('\n', img_th)
#     print(Self_HP)
    return Self_HP

def get_Self_Posture(img):
    img_roi = roi(img, x=240, x_w=290, y=234, y_h=234+1)[0]
    retval, img_th = cv2.threshold(img_roi, 100, 255, cv2.THRESH_TOZERO)
    img_th = np.reshape(img_th, (img_roi.shape))
    
    if int(img_th[0]) - int(img_th[1]) > 15:
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

def get_Target_HP(img):
    img_roi = roi(img, x=29, x_w=130, y=25, y_h=25+1)[0]
    retval, img_th = cv2.threshold(img_roi, 40, 255, cv2.THRESH_TOZERO)
    retval, img_th = cv2.threshold(img_th, 80, 255, cv2.cv2.THRESH_TOZERO_INV)
    img_th = np.reshape(img_th, (img_roi.shape))
    Target_HP = get_value(img_th)
#     print('\n', img_th)
#     print(Target_HP)
    return Target_HP

def get_Target_Posture(img):
    img_roi = roi(img, x=240, x_w=327, y=17, y_h=17+1)[0]
    retval, img_th = cv2.threshold(img_roi, 100, 255, cv2.THRESH_TOZERO)
    img_th = np.reshape(img_th, (img_roi.shape))
    
    if int(img_th[0]) - int(img_th[1]) > 15:
        if img_th[1] in range(100, 125) and img_th[0] in range(175, 222):
            Target_Posture = get_value(img_th)
        elif img_th[1] in range(125, 210) and img_th[0] in range(190, 250):
            Target_Posture = get_value(img_th)
        else:
            Target_Posture = 0
    else:
        Target_Posture = 0
#    print('\n', img_th)
#     print(Target_Posture)
    return Target_Posture

# ---*---

def get_status(img):
    return [get_Self_HP(img), get_Self_Posture(img), get_Target_HP(img), get_Target_Posture(img)]