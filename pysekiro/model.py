import os

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(tf.config.experimental.get_device_details(gpus[0])['device_name'])

def identity_block(input_tensor,out_dim):
    conv1 = tf.keras.layers.Conv2D(out_dim // 4, kernel_size=1, padding="SAME", activation=tf.nn.relu)(input_tensor)
    conv2 = tf.keras.layers.BatchNormalization()(conv1)
    conv3 = tf.keras.layers.Conv2D(out_dim // 4, kernel_size=3, padding="SAME", activation=tf.nn.relu)(conv2)
    conv4 = tf.keras.layers.BatchNormalization()(conv3)
    conv5 = tf.keras.layers.Conv2D(out_dim, kernel_size=1, padding="SAME")(conv4)
    out = tf.keras.layers.Add()([input_tensor, conv5])
    out = tf.nn.relu(out)
    return out
def MODEL(width, height, frame_count, outputs, model_weights=None):

    input_xs = tf.keras.Input(shape=[width, height, frame_count])

    out_dim = 32
    conv_1 = tf.keras.layers.Conv2D(filters=out_dim,kernel_size=3,padding="SAME",activation=tf.nn.relu)(input_xs)

    out_dim = 32
    identity_1 = tf.keras.layers.Conv2D(filters=out_dim, kernel_size=3, padding="SAME", activation=tf.nn.relu)(conv_1)
    identity_1 = tf.keras.layers.BatchNormalization()(identity_1)
    for _ in range(3):
        identity_1 = identity_block(identity_1,out_dim)

    out_dim = 32
    identity_2 = tf.keras.layers.Conv2D(filters=out_dim, kernel_size=3, padding="SAME", activation=tf.nn.relu)(identity_1)
    identity_2 = tf.keras.layers.BatchNormalization()(identity_2)
    for _ in range(3):
        identity_2 = identity_block(identity_2,out_dim)

    flat = tf.keras.layers.Flatten()(identity_2)
    dense = tf.keras.layers.Dense(16,activation=tf.nn.relu)(flat)
    dense = tf.keras.layers.BatchNormalization()(dense)

    logits = tf.keras.layers.Dense(outputs,activation=tf.nn.softmax)(dense)

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