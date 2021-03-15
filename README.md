## 在《只狼：影逝二度》中用深度强化学习训练Agent

#### If you need the English version, please send an issue.  

![demo.jpg](https://raw.githubusercontent.com/ricagj/pysekiro/main/imgs/adjustment_02.png)  

## 最新说明 

#### 准备更新

在目前的代码中（[train.py](https://github.com/ricagj/pysekiro_with_RL/blob/main/pysekiro/train.py)）  
~~~python
def getSARS_(self):

    # 第二个轮回开始
    #   1. 原本的 新状态S' 变成 状态S
    #   2. 由 状态S 选取 动作A
    #   3. 观测并获取 新状态S'，并计算 奖励R
    # 进入下一个轮回

    self.action = self.sekiro_agent.choose_action(self.screen, self.train)    # 选取 动作A

    # 延迟观测新状态，为了能够观测到状态变化
    time.sleep(0.1)

    next_screen = get_screen()    # 观测 新状态
    status = get_status(next_screen)
    self.status_info = status[4]    # 状态信息
    
    self.reward = self.sekiro_agent.reward_system.get_reward(status[:4], self.action)    # 计算 奖励R
    self.next_screen = cv2.resize(roi(next_screen, x, x_w, y, y_h), (WIDTH, HEIGHT))    # 获取 新状态S'

    self.learn()

    # ----- 下一个轮回 -----

    # 保证 状态S 和 新状态S' 连续
    self.screen = self.next_screen    # 状态S
~~~
**获取 状态S 动作A 奖励R 新状态S' 时，当前数据的新状态S'就是下一个数据的状态S，这样的话，存储经验时每条经验之间的状态都是连续的**
~~~python
# ----- 下一个轮回 -----

# 保证 状态S 和 新状态S' 连续
self.screen = self.next_screen    # 状态S
~~~
但是，我想了一下发现这不是必要的，因为经验回放会打乱数据的相关性，那这么做除了少调用一次抓取屏幕的函数以为就没什么用了。  
而且有一个很大的问题：下一轮回动作A的选取时用到的状态不是最新的，而是它的上一个轮回的新状态，可能会出现它的反应慢半拍的情况。  
这个问题很好解决，在选取动作前再调用一次抓取屏幕就好了，这样就保证动作读取时所依据的状态是最新的。  

#### 最新更新

更新读取架势的算法，采用Canny边缘检测 
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