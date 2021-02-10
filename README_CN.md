## 用强化学习玩《只狼：影逝二度》

<p align="center">
    <a href="https://github.com/ricagj/pysekiro_with_RL/blob/main/README.md">English</a>
    | 
    <a>中文</a>
</p>

# 参考
https://github.com/Sentdex/pygta5  
https://github.com/analoganddigital/DQN_play_sekiro  

# 说明

None

# 开发日志

<a href="https://github.com/ricagj/pysekiro_with_RL/blob/main/Development_log/README_CN.md">开发日志</a>

# 准备

#### 游戏设置

- 打开游戏 《只狼：影逝二度》
    - 设定
        - 按键设置
            - 攻击动作
                - 攻击    键盘    J
                - 防御    键盘    K
        - 图像设定
            - 屏幕模式    窗口
            - 屏幕分辨率    1280 x 720

#### 安装 Anaconda3
https://www.anaconda.com/

#### 创建虚拟环境和安装依赖
~~~shell
conda create -n pysekiro python=3.8
conda activate pysekiro
conda install pandas
conda install pywin32
pip install opencv-python>=4.0
pip install tensorflow>=2.0

conda install -c conda-forge jupyterlab
pip install gym
~~~