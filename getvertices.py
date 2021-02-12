import os

import cv2
import numpy as np

def roi(img, x, x_w, y, y_h):
    return img[y:y_h, x:x_w]

def GrabCut_ROI(img, vertices):
    """
    roi内不变，roi外全黑
    Roi inside unchanged, roi outside all black.
    """
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global vertices
    if event == cv2.EVENT_LBUTTONDOWN:
        vertices.append([x, y])
        try:
        	cv2.imshow("window", img)
        except NameError:
        	pass
    return vertices

def standardize(vertices):
    """
    变成矩形
    Become rectangular
    """
    x = min(vertices[0][0], vertices[1][0])
    x_w = max(vertices[2][0], vertices[3][0])
    y = min(vertices[1][1], vertices[2][1])
    y_h = max(vertices[0][1], vertices[3][1])
    vertices = [[x, y_h], [x, y], [x_w, y], [x_w, y_h]]
    return x, x_w, y, y_h, vertices

def get_vertices(img):
    
    global vertices
    vertices = []

    print('Press "ESC" to quit. ') # 按ESC键离开。
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("window", on_EVENT_LBUTTONDOWN)
    while(1):
        cv2.imshow("window", img)
        if cv2.waitKey(0)&0xFF==27:
            break
    cv2.destroyAllWindows()

    if len(vertices) != 4:
        print("vertices number not match")
        return -1

    x, x_w, y, y_h, vertices = standardize(vertices)

    dst = GrabCut_ROI(img, [np.array(vertices)])

    cv2.imshow('img', img)
    cv2.imshow('dst', dst)
    cv2.imshow('roi(img)', roi(img, x, x_w, y, y_h))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print()
    print(f'x={x}, x_w={x_w}, y={y}, y_h={y_h}')
    print(f'vertices={vertices}')
    print()

# ---*---

def demo_01():
    img = cv2.imread("demo.png", 0)
    get_vertices(img)

def demo_02():
    boss = 'Genichiro_Ashina' # 苇名弦一郎
    data = np.load(os.path.join('The_battle_memory', boss, f'training_data-81.npy'), allow_pickle=True)
    
    n = 975
    img =data[n][0]

    get_vertices(img)
    
# ---*---

if __name__ == '__main__':
    # demo_01()
    demo_02()