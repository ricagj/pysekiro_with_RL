## Using Reinforcement Learning to Play 《Sekiro™ Shadows Die Twice》

<p align="center">
    <a>English</a>
    | 
    <a href="https://github.com/ricagj/pysekiro_with_RL/blob/main/README.md">中文</a>
</p>

# Reference
https://github.com/Sentdex/pygta5  
https://github.com/analoganddigital/sekiro_tensorflow  
https://github.com/analoganddigital/DQN_play_sekiro  
https://github.com/ZhiqingXiao/rl-book/blob/master/chapter10_atari/BreakoutDeterministic-v4_tf.ipynb  
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/5_Deep_Q_Network   

# Description

None

# Preparation

#### Game settings
- Open the game 《Sekiro™ Shadows Die Twice》
    - Options
        - Sound and Display
            - Adjust Brightness
                - set 5
        - Key Config
            - Movement
                - Move Forward
                    - set W
                - Move Back
                    - set S
                - Move Left
                    - set A
                - Move Right
                    - set D
                - Step Dodge, (hold) Sprint
                    - set .Shift
                - Jump
                    - set Space
            - Camera Controls
                - Camera Reset/Lock On
                    - set Y
            - Attack Action
                - Attack
                    - set J
                - Deflect, (Hold) Gurad
                    - set K
        - Graphics Options
            - Screen Mode
                - set Windowed
            - Screen Resolution
                - set 1280 x 720
            - Automatic Rendering Adjustment
                - set Off
            Quality Settings
                - set Low
- Place the game window on the upper left
![example_01](./imgs/example_01.png)  

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