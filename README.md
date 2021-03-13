## 在《只狼：影逝二度》中用深度强化学习训练Agent

#### If you need the English version, please send an issue.  

![demo.jpg](https://github.com/ricagj/pysekiro_with_RL/blob/main/demo.jpg?raw=true)  

## 最新说明

1. 接下来进入训练时期，期间不会再像之前那样，代码频繁更新，两三天就大变样。
2. 之后的每次修改代码，都会等那次训练的记忆容量填满之后才修改。
	- replay_memory_size = 75000，一般情况下记忆量需要达到这个量才允许修改代码
	- 出现bug除外
	- 训练效果异常差除外

[训练历史](https://github.com/ricagj/pysekiro/blob/main/train_history.ipynb)

#### 本次更新

- 记忆容量
	- replay_memory_size = 75000 **->** 22500
- 开始经验回放时存储的记忆量
	- replay_start_size  = 5000 **->** 500
- 探索衰减率
	- epsilon_decrease_rate = 0.9999 **->** 0.9988
- 最终探索率
	- min_epsilon = 0.1 **->** 0.3
- 每回合的周期
	- t = 0.2 **->** 0.25
- 
- 增大攻防触发的概率，减小垫跳触发的概率
- 给对攻击或者防御增加额外奖励，条件是造成目标的状态改变
- 在选取动作后观测新状态前增加延迟，为了能够观测到状态变化

## 降低训练标准

1. 默认在作弊模式下训练
2. 目标是苇名弦一郎前两阶段（第三阶段与前两阶段差异较大，对训练不友好）

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