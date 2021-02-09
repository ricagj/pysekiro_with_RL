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

# 准备

#### 先安装 Anaconda3
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