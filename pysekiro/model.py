import os

import tensorflow as tf

def MODEL(width, height, frame_count, outputs, lr, load_weights_path=None):

    input_xs = tf.keras.Input(shape=[width, height, frame_count])

    out_dim = 16
    conv_1 = tf.keras.layers.Conv2D(filters=out_dim,kernel_size=5,padding="same",activation=tf.nn.relu)(input_xs)
    stand1 = tf.keras.layers.BatchNormalization(axis= 1)(conv_1)

    out_dim = 16
    conv_2 = tf.keras.layers.Conv2D(filters=out_dim,kernel_size=3,padding="same",activation=tf.nn.relu)(stand1)
    pooling2 = tf.keras.layers.MaxPool2D(pool_size= [2, 2], strides= [2, 2], padding= 'valid')(conv_2)
    stand2 = tf.keras.layers.BatchNormalization(axis= 1)(pooling2)

    out_dim = 16
    conv_3 = tf.keras.layers.Conv2D(filters=out_dim,kernel_size=3,padding="same",activation=tf.nn.relu)(stand2)
    pooling3 = tf.keras.layers.AveragePooling2D(pool_size= [2, 2], strides= [2, 2], padding= 'valid')(conv_3)
    stand3 = tf.keras.layers.BatchNormalization(axis= 1)(pooling3)

    flat = tf.keras.layers.Flatten()(stand3)

    dense1 = tf.keras.layers.Dense(16,activation=tf.nn.relu)(flat)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)

    dense2 = tf.keras.layers.Dense(16,activation=tf.nn.relu)(flat)
    dense2 = tf.keras.layers.BatchNormalization()(dense2)

    output = tf.keras.layers.Dense(outputs,activation=tf.nn.softmax)(dense2)

    model = tf.keras.Model(inputs=input_xs, outputs=output)

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