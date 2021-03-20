import os

import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(tf.config.experimental.get_device_details(gpus[0])['device_name'])

def MODEL(in_depth, in_height, in_width, in_channels, outputs, lr, load_weights_path=None):

    Input = tf.keras.Input(shape=[in_depth, in_height, in_width, in_channels])

    out_dim = 64
    conv_0 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(in_depth, 3, 3), padding='same', activation=tf.nn.relu)(Input)
    pool_0 = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2))(conv_0)

    conv_1 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(1, 1, 1), padding='same', activation=tf.nn.relu)(pool_0)
    conv_2_0 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(1, 3, 3), padding='same', activation=tf.nn.relu)(conv_1)
    conv_2_1 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(3, 1, 1), padding='same', activation=None)(conv_1)
    conv_2 = tf.keras.layers.Add()([conv_2_0, conv_2_1])
    conv_2_relu = tf.nn.relu(conv_2)
    conv_3 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(1, 1, 1), padding='same', activation=None)(conv_2_relu)
    out = tf.keras.layers.Add()([pool_0, conv_3])
    out_relu = tf.nn.relu(out)

    conv_4 = tf.keras.layers.Conv3D(filters=32, kernel_size=(in_depth, 3, 3), padding='same', activation=tf.nn.relu)(out_relu)
    pool_4 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2))(conv_4)

    conv_5 = tf.keras.layers.Conv3D(filters=32, kernel_size=(in_depth//2, 3, 3), padding='same', activation=tf.nn.relu)(pool_4)
    pool_5 = tf.keras.layers.MaxPool3D(pool_size=(3, 2, 2))(conv_5)

    flat = tf.keras.layers.Flatten()(pool_5)

    dense = tf.keras.layers.Dense(16,activation=tf.nn.relu)(flat)
    dense = tf.keras.layers.BatchNormalization()(dense)

    output = tf.keras.layers.Dense(outputs,activation=tf.nn.softmax)(dense)

    model = tf.keras.Model(inputs=Input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr),
        loss=tf.keras.losses.MeanSquaredError(),
    )

    if load_weights_path:
        if os.path.exists(load_weights_path):
            model.load_weights(load_weights_path)
            print('Load ' + load_weights_path)
        else:
            print('Nothing to load')
    return model

# ---*---

def main():
    in_depth = 6
    in_height = 400
    in_width = 400
    in_channels = 3
    outputs = 5
    lr = 0.01
    model = MODEL(in_depth, in_height, in_width, in_channels, outputs, lr)
    model.summary()

    tensorboard = tf.keras.callbacks.TensorBoard()

    model.fit(
        np.zeros((in_depth, in_height, in_width, in_channels)).reshape(-1, in_depth, in_height, in_width, in_channels),
        np.array([[0, 0, 0, 0, 1]]),
        verbose=0,
        callbacks=[tensorboard]
    )

# ---*---

if __name__ == '__main__':
    main()