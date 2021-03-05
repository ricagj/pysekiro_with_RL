import os

import numpy as np

from pysekiro.get_vertices import roi
from pysekiro.model import MODEL

ROI_WIDTH = 100
ROI_HEIGHT = 200
FRAME_COUNT = 1

x=190
x_w=290
y=30
y_h=230

n_action = 5

def train(
    target,
    start=1,
    end=1,
    batch_size=8,
    epochs=1,
    model_weights=None
    ):

    model = MODEL(ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT,
        outputs = n_action,
        model_weights = model_weights
    )
    model.summary()

    model_weights = 'dl_weights.h5'

    for i in range(start, end+1):

        filename = f'training_data-{i}.npy'
        data_path = os.path.join('The_battle_memory', target, filename)

        if os.path.exists(data_path):
            data = np.load(data_path, allow_pickle=True)
            print(filename, 'Total data volume：', len(data))

            X = np.array([roi(i[0], x, x_w, y, y_h) for i in data]).reshape(-1, ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT)
            Y = np.array([i[1] for i in data])

            model.fit(X, Y, batch_size=batch_size, epochs=epochs, verbose=1)
            model.save_weights(model_weights)
        else:
            print(f'{filename} does not exist ')