# 在《只狼：影逝二度》中用深度强化学习训练Agent

<p align="center">
    <a href="https://github.com/ricagj/pysekiro_with_RL/blob/main/README_EN.md">English</a>
    | 
    <a>中文</a>
</p>

![demo.png](https://github.com/ricagj/pysekiro/blob/main/imgs/demo.png?raw=true)  

## 快速开始

[快速开始](https://github.com/ricagj/pysekiro_with_RL/blob/main/Quick_start.ipynb)  
[如何训练](https://github.com/ricagj/pysekiro_with_RL/blob/main/How_is_it_trained.ipynb)  

## 项目结构

[了解项目基础部分是如何工作的](https://github.com/ricagj/pysekiro_with_RL/blob/main/How_it_works.ipynb)  

- Data_quality (存放收集数据时记录的reward曲线)
    - Genichiro_Ashina （苇名弦一郎）
        - training_data-1.png （第一个战斗数据的reward曲线）
- 
- The_battle_memory
    - Genichiro_Ashina （苇名弦一郎）
        - training_data-1.npy （第一个战斗数据）
- 
- pysekiro
    - \__init__.py
    - 
    - adjustment.py (游戏窗口校准)
    - 
    - collect_data.py (收集数据)
    - 
    - Agent.py (DQN)
    - model.py （模型定义）
    - 
    - grab_screen.py (屏幕图像抓取)
    - get_vertices.py (顶点位置获取)
    - get_status.py (状态获取)
    - 
    - get_keys.py (捕获键盘的按键)
    - direct_keys.py (控制键盘的按键)
    - actions.py (动作控制)
- 
- learn_offline.py (离线学习)
- learn_online.py (在线学习\测试模型)

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