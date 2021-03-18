## 在《只狼：影逝二度》中用深度强化学习训练Agent

#### If you need the English version, please send an issue.  

![demo.jpg](https://raw.githubusercontent.com/ricagj/pysekiro/main/imgs/adjustment_02.png)  

## 最新说明 

这两天了解了下视频行为识别，然后试着搭了个P3D模型，  
![P3D](https://pic4.zhimg.com/v2-cb8f22fc6ae9965ed09a9c85e6453d77_r.jpg)  
[论文地址](https://arxiv.org/abs/1711.10305)  
~~结果发现约5000w个参数。。。对不起，我的**GTX970m**说我不配用这个模型。~~  
对不起，是我蠢，把池化层给忘记了

## P3D 代码实现

~~~python
def P3D_A(input_tensor, out_dim):
    conv_1 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(1, 1, 1), padding='same', activation=tf.nn.relu)(input_tensor)
    conv_2 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(1, 3, 3), padding='same', activation=tf.nn.relu)(conv_1)
    conv_3 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(3, 1, 1), padding='same', activation=tf.nn.relu)(conv_2)
    conv_4 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(1, 1, 1), padding='same', activation=None)(conv_3)
    out    = tf.keras.layers.Add()([input_tensor, conv_4])
    out    = tf.nn.relu(out)
    return out

def P3D_B(input_tensor, out_dim):
    conv_1 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(1, 1, 1), padding='same', activation=tf.nn.relu)(input_tensor)
    conv_2 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(1, 3, 3), padding='same', activation=tf.nn.relu)(conv_1)
    conv_3 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(3, 1, 1), padding='same', activation=None)(conv_1)
    add_1  = tf.keras.layers.Add()([conv_2, conv_3])
    add_1  = tf.nn.relu(add_1)
    conv_4 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(1, 1, 1), padding='same', activation=None)(add_1)
    out    = tf.keras.layers.Add()([input_tensor, conv_4])
    out    = tf.nn.relu(out)
    return out

def P3D_C(input_tensor, out_dim):
    conv_1 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(1, 1, 1), padding='same', activation=tf.nn.relu)(input_tensor)
    conv_2 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(1, 3, 3), padding='same', activation=tf.nn.relu)(conv_1)
    conv_3 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(3, 1, 1), padding='same', activation=None)(conv_2)
    add_1  = tf.keras.layers.Add()([conv_2, conv_3])
    add_1  = tf.nn.relu(add_1)
    conv_4 = tf.keras.layers.Conv3D(filters=out_dim, kernel_size=(1, 1, 1), padding='same', activation=None)(add_1)
    out    = tf.keras.layers.Add()([input_tensor, conv_4])
    out    = tf.nn.relu(out)
    return out
~~~

## 降低训练标准

1. 默认在作弊模式下训练
2. 全程夜叉戮之降临
3. 目标是苇名弦一郎前两阶段（第三阶段与前两阶段差异较大，对训练不友好）

## 快速开始

[快速开始](https://github.com/ricagj/pysekiro_with_RL/blob/main/Quick_start.ipynb)  

## 项目结构

[了解项目基础部分是如何工作的](https://github.com/ricagj/pysekiro_with_RL/blob/main/How_it_works.ipynb)  

- pysekiro
    - img_tools
        - \__init__.py
        - adjustment.py (游戏窗口校准)
        - get_status.py (状态获取)
        - get_vertices.py (顶点位置获取)
        - grab_screen.py (屏幕图像抓取)
    - key_tools
        - \__init__.py
        - actions.py (动作控制)
        - direct_keys.py (控制键盘的按键)
        - get_keys.py (捕获键盘的按键)
    - \__init__.py
    - Agent.py (DQN)
    - model.py （模型定义）
    - train.py (训练\测试模型)

## 准备

#### 安装 Anaconda3

https://www.anaconda.com/  

#### 创建虚拟环境和安装依赖

~~~shell
conda create -n pysekiro python=3.8
conda activate pysekiro
conda install pandas
conda install matplotlib
conda install pywin32
pip install opencv-python>=4.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorflow>=2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install -c conda-forge jupyterlab
~~~

## 参考
https://github.com/Sentdex/pygta5  
https://github.com/analoganddigital/sekiro_tensorflow  
https://github.com/analoganddigital/DQN_play_sekiro  
https://github.com/ZhiqingXiao/rl-book/blob/master/chapter10_atari/BreakoutDeterministic-v4_tf.ipynb  
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/5_Deep_Q_Network  