# -*- coding:utf-8 -*-

import os

import cv2
import numpy as np

from pysekiro.Agent import resnet
from pysekiro.get_vertices import roi

# ---*---

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(tf.config.experimental.get_device_details(gpus[0])['device_name'])

# ---*---

# x, x_w, y, y_h. 这些数据获取自 getvertices.py
# x, x_w, y, y_h. Get this data from getvertices.py
x=190
x_w=290
y=30
y_h=230

ROI_WIDTH = x_w - x
ROI_HEIGHT = y_h - y
FRAME_COUNT = 1

MODEL_WEIGHTS_NAME = 'sekiro_weights.h5'
model = resnet(ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT, output=5)
# model.summary()

# ---*---

def train(boss, start, end):
    global model
    for i in range(start, end+1):

        filename = f'training_data-{i}.npy'

        # 加载数据
        # Load data
        train_data = np.load(os.path.join('The_battle_memory', boss, filename), allow_pickle=True)
        print(filename, 'Total data volume：', len(train_data))

        # 拆分数据和标签
        # Split data and labels
        X = np.array([roi(i[0], x, x_w, y, y_h) for i in train_data]).reshape(-1, ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT)
        Y = np.array([i[1] for i in train_data])

        # 用 TensorBoard 可视化训练过程
        # Visualize the training process with TensorBoard
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir = os.path.join('logs', filename[:-4]),
            histogram_freq = 1,
            update_freq='batch'
        )

        # 模型训练
        # Model training
        model.fit(X, Y, batch_size=64, epochs=2, verbose=1, callbacks=[tensorboard])

        # 保存模型
        # Save model
        model.save_weights(MODEL_WEIGHTS_NAME)

# ---*---

# def evaluate(boss, start, end):
#     global model
#     for i in range(start, end+1):

#         filename = f'training_data-{i}.npy'

#         # 加载数据
#         # Load data
#         test_data = np.load(os.path.join('The_battle_memory', boss, filename), allow_pickle=True)
#         print(filename, 'Total data volume：', len(test_data))

#         # 拆分数据和标签
#         # Split data and labels
#         X = np.array([roi(i[0], x, x_w, y, y_h) for i in test_data]).reshape(-1, ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT)
#         Y = np.array([i[1] for i in test_data])

#         # 用 TensorBoard 可视化训练过程
#         # Visualize the training process with TensorBoard
#         tensorboard = tf.keras.callbacks.TensorBoard(
#             log_dir = os.path.join('evaluate_logs', filename[:-4]),
#             histogram_freq = 1,
#             update_freq='batch'
#         )

#         model.evaluate(X, Y, batch_size=1, callbacks=[tensorboard])

# ---*---

boss1 = 'Genichiro_Ashina' # 苇名弦一郎
# boss2 = 'Inner_Genichiro'  # 心中的弦一郎
# boss3 = 'Inner_Isshin'     # 心中的一心
# boss4 = 'Isshin,_the_Sword_Saint' # 剑圣 苇名一心

train(boss1, start=1, end=101)
# evaluate(boss1, start=91, end=101)