## 在《只狼：影逝二度》中用深度强化学习训练Agent

#### If you need the English version, please send an issue.  

![demo.jpg](https://github.com/ricagj/pysekiro_with_RL/blob/main/demo.jpg?raw=true)  

## **正在调试**

## 快速开始

[快速开始](https://github.com/ricagj/pysekiro_with_RL/blob/main/Quick_start.ipynb)  

新增作弊模式  
游戏对它来说太难了，进入游戏5秒都撑不过，所以尝试一下让它作弊，在不计生命值情况下训练  
**注：此模式下奖励的计算，不计生命值变化，只计架势变化。**

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