import tensorflow as tf

def identity_block(input_tensor,out_dim):
    conv1 = tf.keras.layers.Conv2D(out_dim // 4, kernel_size=1, padding="SAME", activation=tf.nn.relu)(input_tensor)
    conv2 = tf.keras.layers.BatchNormalization()(conv1)
    conv3 = tf.keras.layers.Conv2D(out_dim // 4, kernel_size=3, padding="SAME", activation=tf.nn.relu)(conv2)
    conv4 = tf.keras.layers.BatchNormalization()(conv3)
    conv5 = tf.keras.layers.Conv2D(out_dim, kernel_size=1, padding="SAME")(conv4)
    out = tf.keras.layers.Add()([input_tensor, conv5])
    out = tf.nn.relu(out)
    return out
def resnet(width, height, frame_count, output):

    input_xs = tf.keras.Input(shape=[width, height, frame_count])
    
    out_dim = 32 # 64
    conv_1 = tf.keras.layers.Conv2D(filters=out_dim,kernel_size=3,padding="SAME",activation=tf.nn.relu)(input_xs)

    """-------- 1 ----------"""
    out_dim = 16 # 64
    identity_1 = tf.keras.layers.Conv2D(filters=out_dim, kernel_size=3, padding="SAME", activation=tf.nn.relu)(conv_1)
    identity_1 = tf.keras.layers.BatchNormalization()(identity_1)
    for _ in range(2): # 3
        identity_1 = identity_block(identity_1,out_dim)

    """-------- 2 ----------"""
    out_dim = 16 # 128
    identity_2 = tf.keras.layers.Conv2D(filters=out_dim, kernel_size=3, padding="SAME", activation=tf.nn.relu)(identity_1)
    identity_2 = tf.keras.layers.BatchNormalization()(identity_2)
    for _ in range(2): # 4
        identity_2 = identity_block(identity_2,out_dim)

#     """-------- 3 ----------"""
#     out_dim = 256
#     identity_3 = tf.keras.layers.Conv2D(filters=out_dim, kernel_size=3, padding="SAME", activation=tf.nn.relu)(identity_2)
#     identity_3 = tf.keras.layers.BatchNormalization()(identity_3)
#     for _ in range(6):
#         identity_3 = identity_block(identity_3,out_dim)

#     """-------- 4 ----------"""
#     out_dim = 512
#     identity_4 = tf.keras.layers.Conv2D(filters=out_dim, kernel_size=3, padding="SAME", activation=tf.nn.relu)(identity_3)
#     identity_4 = tf.keras.layers.BatchNormalization()(identity_4)
#     for _ in range(3):
#         identity_4 = identity_block(identity_4,out_dim)

    flat = tf.keras.layers.Flatten()(identity_2) # identity_4
    flat = tf.keras.layers.Dropout(0.5)(flat) # 0.217
    dense = tf.keras.layers.Dense(32,activation=tf.nn.relu)(flat) # 2048
    dense = tf.keras.layers.BatchNormalization()(dense)
    
    logits = tf.keras.layers.Dense(output,activation=tf.nn.softmax)(dense)
    
    model = tf.keras.Model(inputs=input_xs, outputs=logits)
    
    model.compile(
        optimizer = tf.keras.optimizers.Nadam(),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = [
            'accuracy'
        ]
    )
    
    return model

# ---*---

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(tf.config.experimental.get_device_details(gpus[0])['device_name'])

    # x, x_w, y, y_h. 这些数据获取自 getvertices.py
    # x, x_w, y, y_h. Get this data from getvertices.py
    x=190
    x_w=290
    y=30
    y_h=230

    ROI_WIDTH = x_w - x
    ROI_HEIGHT = y_h - y
    FRAME_COUNT = 1

    MODEL_NAME = 'sekiro.h5'
    model = resnet(ROI_WIDTH, ROI_HEIGHT, FRAME_COUNT, output=5)

    model.summary()