from pysekiro.get_vertices import roi

# ---*---

def get_HP(target_img):
    count = 0
    
    if target_img[0] == 0 or target_img[1] == 0:
        return count
    
    for i in range(len(target_img)-1):
        cur_pixel = int(target_img[i])
        next_pixel = int(target_img[i+1])
        if abs(cur_pixel - next_pixel) > 20 or cur_pixel < 40 or cur_pixel > 80:
            break
        count += 1
    return count

def get_Posture(target_img):
    count = 0
    
    if target_img[0] == 0 or target_img[1] == 0:
        return count
    
    for i in range(len(target_img)-1):
        cur_pixel = int(target_img[i])
        next_pixel = int(target_img[i+1])
        if abs(cur_pixel - next_pixel) > 20 or cur_pixel < 100:
            break
        count += 1
    return count

# ---*---

def get_Sekiro_HP(img):
    img_roi = roi(img, x=29, x_w=182, y=244, y_h=246)[0]
    Sekiro_HP = get_HP(img_roi)
#     print('\n\n', img_roi, Sekiro_HP)
    return Sekiro_HP

def get_Sekiro_Posture(img):
    img_roi = roi(img, x=241, x_w=290, y=233, y_h=235)[0]
    Sekiro_Posture = get_Posture(img_roi)
#     print('\n\n', img_roi, Sekiro_Posture)
    return Sekiro_Posture

def get_Boss_HP(img):
    img_roi = roi(img, x=29, x_w=129, y=24, y_h=26)[0]
    Boss_HP = get_HP(img_roi)
#     print('\n\n', img_roi, Boss_HP)
    return Boss_HP

def get_Boss_Posture(img):
    img_roi = roi(img, x=241, x_w=326, y=16, y_h=18)[0]
    Boss_Posture = get_Posture(img_roi)
#     print('\n\n', img_roi, Boss_Posture)
    return Boss_Posture

# ---*---

def get_status(img):
    return [get_Sekiro_HP(img), get_Sekiro_Posture(img), get_Boss_HP(img), get_Boss_Posture(img)]