## 《只狼：影逝二度》【DoubleDQN】【Conv3D】

<p align="center">
    <a href="https://github.com/ricagj/pysekiro_with_RL/blob/main/README.md">English</a>
    | 
    <a>中文</a>
</p>

![demo.jpg](https://raw.githubusercontent.com/ricagj/pysekiro/main/imgs/adjustment_02.png)  

## 快速开始

[快速开始](https://github.com/ricagj/pysekiro_with_RL/blob/main/CN/Quick_start.ipynb)  

## 项目结构

[了解项目基础部分是如何工作的](https://github.com/ricagj/pysekiro_with_RL/blob/main/CN/How_it_works.ipynb)  

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
    - off_policy.py (异策学习)

## 安装

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