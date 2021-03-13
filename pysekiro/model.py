import os

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(tf.config.experimental.get_device_details(gpus[0])['device_name'])

def identity_block(input_tensor,out_dim):
    conv1 = tf.keras.layers.Conv2D(out_dim // 4, kernel_size=1, padding="SAME", activation=tf.nn.relu)(input_tensor)
    conv2 = tf.keras.layers.BatchNormalization()(conv1)
    conv3 = tf.keras.layers.Conv2D(out_dim, kernel_size=1, padding="SAME")(conv2)
    out = tf.keras.layers.Add()([input_tensor, conv3])
    out = tf.nn.relu(out)
    return out
def MODEL(width, height, frame_count, outputs, load_weights_path=None):

    input_xs = tf.keras.Input(shape=[width, height, frame_count])

    out_dim = 16
    conv_1 = tf.keras.layers.Conv2D(filters=out_dim,kernel_size=3,padding="SAME",activation=tf.nn.relu)(input_xs)

    out_dim = 16
    identity = tf.keras.layers.Conv2D(filters=out_dim, kernel_size=3, padding="SAME", activation=tf.nn.relu)(conv_1)
    identity = tf.keras.layers.BatchNormalization()(identity)
    for _ in range(1):
        identity = identity_block(identity,out_dim)

    flat = tf.keras.layers.Flatten()(identity)
    dense = tf.keras.layers.Dense(16,activation=tf.nn.relu)(flat)
    dense = tf.keras.layers.BatchNormalization()(dense)

    logits = tf.keras.layers.Dense(outputs,activation=tf.nn.softmax)(dense)

    model = tf.keras.Model(inputs=input_xs, outputs=logits)

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.MeanSquaredError(),
    )

    if load_weights_path:
        if os.path.exists(load_weights_path):
            model.load_weights(load_weights_path)
            print('Load ' + load_weights_path)
        else:
            print('Nothing to load')
    return model