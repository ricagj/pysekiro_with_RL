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
def MODEL(width, height, frame_count, outputs, model_weights=None):

    input_xs = tf.keras.Input(shape=[width, height, frame_count])

    out_dim = 16
    conv = tf.keras.layers.Conv2D(filters=out_dim,kernel_size=3,padding="SAME",activation=tf.nn.relu)(input_xs)

    out_dim = 16
    identity = tf.keras.layers.Conv2D(filters=out_dim, kernel_size=3, padding="SAME", activation=tf.nn.relu)(conv)
    identity = tf.keras.layers.BatchNormalization()(identity)
    for _ in range(1):
        identity = identity_block(identity,out_dim)

    flat = tf.keras.layers.Flatten()(identity)
    dense = tf.keras.layers.Dense(16,activation=tf.nn.relu)(flat)
    dense = tf.keras.layers.BatchNormalization()(dense)

    logits = tf.keras.layers.Dense(outputs,activation=None)(dense)

    model = tf.keras.Model(inputs=input_xs, outputs=logits)

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.MeanSquaredError(),
    )

    if model_weights:
        if os.path.exists(model_weights):
            model.load_weights(model_weights)
            print('Load ' + model_weights)
        else:
            print('Nothing to load')
    return model