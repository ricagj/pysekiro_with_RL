## Using Reinforcement Learning to Play 《Sekiro™ Shadows Die Twice》

<p align="center">
    <a>English</a>
    | 
    <a href="https://github.com/ricagj/pysekiro_with_RL/blob/main/README_CN.md">中文</a>
</p>

# Reference
https://github.com/Sentdex/pygta5  
https://github.com/analoganddigital/DQN_play_sekiro  

# Description

None

# Development log

**Pending translation**  
<a href="https://github.com/ricagj/pysekiro_with_RL/blob/main/Development_log/README.md">Development log</a>

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
conda install pywin32
pip install opencv-python>=4.0
pip install tensorflow>=2.0

conda install -c conda-forge jupyterlab
pip install gym
~~~

