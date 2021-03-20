import os
import time

import pandas as pd

from pysekiro.Agent import Sekiro_Agent
from pysekiro.key_tools.get_keys import key_check

# ---*---

class Play_Sekiro_Offline:
    def __init__(
        self,
        lr,
        batch_size,
        load_memory_path,
        save_weights_path,
        load_weights_path=None
    ):
        self.sekiro_agent = Sekiro_Agent(
            lr         = lr,    # 学习率
            batch_size = batch_size,    # 样本抽取数量
            load_weights_path = load_weights_path,    # 指定模型权重参数加载的路径。默认为None，不加载。
            save_weights_path = save_weights_path    # 指定模型权重参数保存的路径。默认为None，不保存。注：只有训练模式，没有测试模式，如果要测试模型的话，就要借用上面的代码
        )

        self.load_memory_path = load_memory_path     # 指定记忆加载的路径。默认为None，不加载。

        self.load_memory()    # 加载经验

    def load_memory(self):
        if os.path.exists(self.load_memory_path):    # 确定经验的存在
            last_time = time.time()
            self.sekiro_agent.replayer.memory = pd.read_json(self.load_memory_path)    # 加载经验
            print(f'Load {self.load_memory_path}. Took {round(time.time()-last_time, 3):>5} seconds.')

            self.sekiro_agent.replayer.count = self.sekiro_agent.replayer.memory.action.count()    # 恢复 "代表经验的数量" 的数值
        else:
            print('No memory to load.')

    def run(self):

        paused = True
        print("Ready!")

        while True:
            keys = key_check()
            if paused:
                if 'T' in keys:
                    paused = False
                    print('\nStarting!')
            else:    # 按 'T' 之后，马上进入下一轮就进入这里
                self.sekiro_agent.learn(verbose=1)    # 直接开始学习

                print(f'\r step:{self.sekiro_agent.step:>6}', end='')

                if 'P' in keys:
                	break

        self.sekiro_agent.save_evaluate_network()    # 学习完毕，保存网络权重