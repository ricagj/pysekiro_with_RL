# 在《只狼：影逝二度》中用深度强化学习训练Agent

<p align="center">
    <a href="https://github.com/ricagj/pysekiro_with_RL/blob/main/README_EN.md">English</a>
    | 
    <a>中文</a>
</p>

![demo.png](https://github.com/ricagj/pysekiro_with_RL/blob/main/demo.pngraw=true)  

# 注：接下来一个版本将会移除离线学习相关代码

# 快速开始

[快速开始](https://github.com/ricagj/pysekiro_with_RL/blob/main/Quick_start.ipynb)  
[如何训练](https://github.com/ricagj/pysekiro_with_RL/blob/main/How_is_it_trained.ipynb)  

# 项目结构

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

# 准备

### 安装 Anaconda3

https://www.anaconda.com/  

### 创建虚拟环境和安装依赖

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

# 存在的问题

1. 本项目只是半成品，目前我未能得到一个令我满意的模型。
2. 这里的Agent并不是游戏中的被操作对象，而是像我们一样的玩家，它也是“通过键盘”来操作被操作对象的。我们在操作的时候，不会时时刻刻都在按键的，而是判断时机，然后在合适的时候按合适的键的。而它不一样，它是每隔零点几秒就必须做出决策然后“按键”。在这个存在后摇的游戏中，有更快的反应固然是好事，但同时也因为这而导致后摇期间的决策是无效的，这些无效的决策也被学习到了。这是今后要解决的难点之一。
3. 还有一个比较麻烦的问题，就是动作空间虽然是离散的，但是动作本身是连续的。最典型的就是长按短按，还有移动。这些无法被模型所学习，也无法被模型所执行（动作控制相关代码的锅）。
4. 当前最重要也最难解决的一个问题应该是动作和奖励不能很好的对应上，在强化学习领域可是大忌。原因跟第2点差不多，不过这次是因为前摇。这个问题我通过控制状态获取频率来试图解决，确实取得了一定成果（一帧一帧观察），但仍然有部分不能完全对应上。这也是今后要解决的难点之一。
5. 离线学习，其实是我期望模型能够利用我收集的数据集来学习以获得基本的对战能力，但目前的学习效果非常差。2020-3-10提交的代码里我保留了训练过程，从训练过程来看，积累了非常多的奖励，按道理应该变得非常厉害了。但事实上，我测试的时候发现，它只学会了苇名抖刀术。。。
6. 奖励设置的问题。奖励也是强化学习一个很重要的问题，它直接影响训练效果。这个还在探索中，所以这部分代码变动比较频繁。

# 参考
https://github.com/Sentdex/pygta5  
https://github.com/analoganddigital/sekiro_tensorflow  
https://github.com/analoganddigital/DQN_play_sekiro  
https://github.com/ZhiqingXiao/rl-book/blob/master/chapter10_atari/BreakoutDeterministic-v4_tf.ipynb  
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/5_Deep_Q_Network  