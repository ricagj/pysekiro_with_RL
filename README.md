## 用强化学习玩《只狼：影逝二度》

<p align="center">
    <a href="https://github.com/ricagj/pysekiro_with_RL/blob/main/README_EN.md">English</a>
    | 
    <a>中文</a>
</p>

![demo.png](https://github.com/ricagj/pysekiro_with_RL/blob/main/imgs/demo.png?raw=true)

# 快速开始

[快速开始](https://github.com/ricagj/pysekiro_with_RL/blob/main/Quick_start.ipynb)  

# 说明

#### 本项目将从三个角度来训练**智能体**

1. 用深度学习从已有的数据中学习
2. 用强化学习从零开始在实战中学习
3. 用强化学习从已有的数据中学习，再在实战中继续学习

[了解训练是如何进行的](https://github.com/ricagj/pysekiro_with_RL/blob/main/How_is_it_trained.ipynb)  

# 项目结构

[了解项目各部分是如何工作的](https://github.com/ricagj/pysekiro_with_RL/blob/main/How_it_works.ipynb)  

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
    - get_keys.py (捕获键盘的按键)
    - direct_keys.py (控制键盘的按键)
    - actions.py (动作控制)
    - 
    - grab_screen.py (屏幕图像抓取)
    - get_vertices.py (顶点位置获取)
    - get_status.py (状态获取)
    - 
    - adjustment.py (游戏窗口校准)
    - 
    - collect_data.py (收集数据)
    - 
    - model.py （模型定义）
    - train_with_dl.py （用深度学习训练）
    - Agent.py (DQN)
- 
- learn_offline.py (离线学习)
- learn_online.py (在线学习\测试模型)

# 准备

#### 安装 Anaconda3

https://www.anaconda.com/  

#### 创建虚拟环境和安装依赖

~~~shell
conda create -n pysekiro python=3.8
conda activate pysekiro
conda install pandas
conda install matplotlib
conda install pywin32
pip install opencv-python>=4.0
pip install tensorflow>=2.0
conda install -c conda-forge jupyterlab
~~~

# 参考
https://github.com/Sentdex/pygta5  
https://github.com/analoganddigital/sekiro_tensorflow  
https://github.com/analoganddigital/DQN_play_sekiro  
https://github.com/ZhiqingXiao/rl-book/blob/master/chapter10_atari/BreakoutDeterministic-v4_tf.ipynb  
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/5_Deep_Q_Network  