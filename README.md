## 《只狼：影逝二度》【DoubleDQN】【Conv3D】

#### If you need the English version, please send an issue.  

![demo.jpg](https://raw.githubusercontent.com/ricagj/pysekiro/main/imgs/adjustment_02.png)  

## 最新说明 

#### 1. 这个项目也差不多要告一段落了，以后就随缘更新了。
#### 2. 不要盲目照搬我的代码，最好自己改一改。目前我并没有训练出令我满意的模型，虽然也有训练时间较短的原因，但是有可能我写的模型根本就是错误的。特别是 model.py 里面的模型，是我自己搭的，可能缺乏一定的理论支撑，所以大家尽量去找一些经过验证的，有效果的模型来替换我的模型，这样或许能在训练的时候有更好的表现。然后，Agent.py 里的代码大家可以放心，这个我参照了好多份类似的代码才写出来的，应该不会有太大的问题。
#### 3. 接下来我会整理一份如何用这套代码在其他游戏训练AI的资料，方便大家训练自己喜欢的游戏的AI。

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
    - Agent.py (DoubleDQN)
    - model.py （模型定义）
    - on_policy.py (同策学习)

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
https://github.com/ZhiqingXiao/rl-book/blob/master/chapter06_approx/MountainCar-v0_tf.ipynb  
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/5_Deep_Q_Network  