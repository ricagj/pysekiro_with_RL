## 在《只狼：影逝二度》中用深度强化学习训练Agent

#### If you need the English version, please send an issue.  

![demo.jpg](https://raw.githubusercontent.com/ricagj/pysekiro/main/imgs/adjustment_02.png)  

# **正在调试**

## 最新说明 

#### 最新更新

1. 换了个相对简单的模型（自己瞎搞的，那些nb的模型我电脑根本带不动）。
2. 删除对选择攻击和防御时附加的额外奖励。
3. 最终探索率调至0.4，探索衰减率由原来的指数衰减更换为线性衰减。
4. 调整各网络的更新频率
5. 上调学习率 0.001 -> 0.01
6. 采用多线程不断的抓取屏幕获取信息

#### 最近更新

1. 更新读取架势的算法，采用Canny边缘检测 
[新旧效果对比](https://github.com/ricagj/pysekiro/blob/main/TEST_get_status.ipynb)  
基本上能稳定读取架势了，但是仍然存在特殊情况。比如Boss生命值为零时，这个时候攻击他，他的架势会瞬间积满，观察Boss的架势条时，会发现有瞬间的红光一闪而过，而且范围超过了架势条的范围。由于检测的范围内像素点全部为255，所以无法找到边缘，输出的结果本来应该是满架势值但实际输出为0。

## 降低训练标准

1. 默认在作弊模式下训练
2. 目标是苇名弦一郎前两阶段（第三阶段与前两阶段差异较大，对训练不友好）

[训练历史](https://github.com/ricagj/pysekiro/blob/main/train_history.ipynb)

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