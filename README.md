## Using Reinforcement Learning to Play 《Sekiro™ Shadows Die Twice》

<p align="center">
    <a>English</a>
    | 
    <a href="https://github.com/ricagj/pysekiro_with_RL/blob/main/README_CN.md">中文</a>
</p>

# Reference
https://github.com/Sentdex/pygta5  
https://github.com/analoganddigital/DQN_play_sekiro  
https://github.com/ZhiqingXiao/rl-book/blob/master/chapter10_atari/BreakoutDeterministic-v4_tf.ipynb  
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/5_Deep_Q_Network  

# **Pending translation**
## **You can choose to read the Chinese version first.**

# Description

None

# Preparation

#### Game settings

- Open the game 《Sekiro™ Shadows Die Twice》
    - Options
        - Key Config
            - Attack Action
                - Attack    keyboard    J
                - Deflect, (Hold) Gurad    keyboard    K
        - Graphics Options
            - Screen Mode    Windowed
            - Screen Resolution    1280 x 720

#### Install Anaconda3
https://www.anaconda.com/

#### Create a virtual environment and install dependencies
~~~shell
conda create -n pysekiro python=3.8
conda activate pysekiro
conda install pandas
conda install matplotlib
conda install pywin32
pip install opencv-python>=4.0
pip install tensorflow>=2.0
~~~

