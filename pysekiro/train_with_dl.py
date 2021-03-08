import os

import cv2
import numpy as np

from pysekiro.get_vertices import roi
from pysekiro.model import MODEL

ROI_WIDTH   = 50
ROI_HEIGHT  = 50
FRAME_COUNT = 3

x   = 140
x_w = 340
y   = 30
y_h = 230

n_action = 5

def train(
    target,
    start=1,
    end=1,
    batch_size=128,
    epochs=1,
    model_weights=None
    ):

    model = MODEL(ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT,
        outputs = n_action,
        model_weights = model_weights
    )
    model.summary()

    model_weights = 'dl_weights.h5'

    # 读取一个数据集训练，然后再读取下一个数据集训练，以此类推
    for i in range(start, end+1):

        filename = f'training_data-{i}.npy'
        data_path = os.path.join('The_battle_memory', target, filename)

        if os.path.exists(data_path):    # 确保数据集存在
        
            # 加载数据集
            data = np.load(data_path, allow_pickle=True)
            print('\n', filename, f'total:{len(data):>5}')

            # 数据集处理成预训练格式
            X = np.array([cv2.resize(roi(i[0], x, x_w, y, y_h), (ROI_WIDTH, ROI_HEIGHT)) for i in data]).reshape(-1, ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT)
            Y = np.array([i[1][:5] for i in data])

            # 训练模型，然后保存
            model.fit(X, Y, batch_size=batch_size, epochs=epochs, verbose=1)
            model.save_weights(model_weights)
        else:
            print(f'{filename} does not exist ')