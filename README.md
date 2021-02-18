## 用强化学习玩《只狼：影逝二度》

<p align="center">
    <a href="https://github.com/ricagj/pysekiro_with_RL/blob/main/README_EN.md">English</a>
    | 
    <a>中文</a>
</p>

# 参考
https://github.com/Sentdex/pygta5  
https://github.com/analoganddigital/sekiro_tensorflow  
https://github.com/analoganddigital/DQN_play_sekiro  
https://github.com/ZhiqingXiao/rl-book/blob/master/chapter10_atari/BreakoutDeterministic-v4_tf.ipynb  
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/5_Deep_Q_Network  

# 说明

#### 本项目将从三个角度来训练只狼
1. 在有收集到一定数据的情况下，单纯只用深度学习训练**智能体只狼**  
2. 在没有数据的情况下，单纯只用强化学习训练**智能体只狼**  
3. 在有收集到一定数据的情况下，先用深度学习训练**智能体只狼**，再把训练出来的模型用强化学习继续训练  

具体见 [教程](https://nbviewer.jupyter.org/github.com/ricagj/pysekiro_with_RL/Tutorial.ipynb)
# 准备

#### 游戏设置
- 打开游戏 《只狼：影逝二度》
    - 设定
        - 显示和声音
            - 亮度调整
                - 设置 5
        - 按键设置
            - 移动
                - 移动 前
                    - 设置 W
                - 移动 后
                    - 设置 S
                - 移动 左
                    - 设置 A
                - 移动 右
                    - 设置 D
                - 垫步、（长按）冲刺
                    - 设置 .Shift
                - 跳跃
                    - 设置 Space
            - 视角操作
                - 重置视角/固定目标
                    - 设置 Y
            - 攻击动作
                - 攻击
                    - 设置 J
                - 防御
                    - 设置 K
        - 图像设定
            - 屏幕模式
                - 窗口
            - 屏幕分辨率
                - 1280 x 720
            - 自动绘图调整
                - OFF
            - 质量设定
                - 低
- 将游戏窗口放在左上方  
![example_01](./imgs/example_01.png)  

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