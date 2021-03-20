## 《只狼：影逝二度》【DoubleDQN】【Conv3D】

#### If you need the English version, please send an issue.  

![demo.jpg](https://raw.githubusercontent.com/ricagj/pysekiro/main/imgs/adjustment_02.png)  

## **新版本已上线**

## 最新说明 

1. 其实整个代码的变化不大，只不过是二维变三维（多了时间序列），其他代码适配新版本的变化而已。
2. 模型已经完成了从二维卷积到三维卷积的升级，但是由于我电脑的配置（GTX970m 3G显存）不够，所以模型的结构被我一改再改，现在勉强能跑。但能跑归能跑，我训练了一回（8000步）发现打到后面还行，基本上能防御，也不会过于依赖苇名抖刀术，就是不太愿意攻击。如果你的电脑配置足够好，你应该试着自己调整模型，说不定调了一下训练效果就翻几倍了。
3. 存储经验的流程
![存储经验部分代码流程图](https://github.com/ricagj/pysekiro_with_RL/blob/main/flow_chart.png)  
4. 所谓同策学习，就是边决策边学习，学习者也是决策者；所谓异策学习，就是可以从自己历史的经验中学习，也可以从别人的经验中学习，学习者不一定是决策者。目前提供的代码只有同策学习的，过会补上异策学习的。我已经设置了结束学习时保存经验，所以到时可以直接利用保存的经验来进行异策学习。
5. 更新完异策学习的代码，这个项目也差不多要告一段落了，嘛，原因有各种各样的，比如电脑配置跟不上啦，还有别的项目要做啦，考研备考的进度要赶一赶啦，还有就是想换个简单点的游戏试一试啦（特别是不需要那么多GPU显存的游戏）。当然，这个项目也不是以后都不更新了，等考完研，换了电脑，我会卷土重来的。

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