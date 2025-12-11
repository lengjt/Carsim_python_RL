import gym
from gym import spaces
import numpy as np
import matlab.engine
import os
import time


class SimulinkGymEnv(gym.Env):
    """
    Simulink 模型交互的标准 Gym 环境封装
    """

    def __init__(self, model_name='Pendulum_Model', dt=0.05, stop_time=20.0):
        super(SimulinkGymEnv, self).__init__()

        # 1. 配置参数
        self.model_name = model_name
        self.dt = dt
        self.stop_time = stop_time

        # 2. 定义动作空间 (Action Space)
        # 例如：倒立摆是 1 维控制 (力矩)，范围 [-2, 2]
        # 如果是车辆模型，请修改 shape=(2,) 并调整 low/high
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)

        # 3. 定义观测空间 (Observation Space)
        # 例如：[cos(theta), sin(theta), dot_theta] 维度为 3
        # 请根据您 Simulink 输出的 obs 维度修改 shape
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # 4. 启动 MATLAB Engine
        print("正在启动 MATLAB Engine，请稍候...")
        # 连接到已有的 MATLAB 窗口，无需等待启动，直接就能交互
        names = matlab.engine.find_matlab()
        if names:
            self.eng = matlab.engine.connect_matlab(names[0])
        else:
            self.eng = matlab.engine.start_matlab()  # 如果没找到再启动新的
        print("MATLAB Engine 启动成功！")
        self.eng.desktop(nargout=0) # 如果想看界面，取消注释

        # 加载模型
        model_path = os.path.abspath('.')
        self.eng.cd(model_path, nargout=0)
        self.eng.load_system(self.model_name)
        print(f"Simulink 模型 '{self.model_name}' 加载成功。")

        self.current_pause_time = 0.0

    def reset(self):
        """
        重置环境：停止当前仿真 -> 设置初始参数 -> 启动仿真 -> 获取初始状态
        """
        # 停止之前的仿真
        try:
            status = self.eng.get_param(self.model_name, 'SimulationStatus')
            if status != 'stopped':
                self.eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
        except:
            pass

        # 重置时间
        self.current_pause_time = self.dt

        # 设置仿真参数
        # 注意：多给一点时间防止刚好到点停止导致拿不到数据
        self.eng.set_param(self.model_name, 'StopTime', str(self.stop_time + self.dt), nargout=0)
        self.eng.set_param(self.model_name + '/pause_time', 'Value', str(self.current_pause_time), nargout=0)

        # 如果有动作输入的初始值，也建议重置，例如设为0
        self.eng.set_param(self.model_name + '/action_input', 'Value', '0', nargout=0)

        # 启动仿真
        self.eng.set_param(self.model_name, 'SimulationCommand', 'start', nargout=0)

        # 等待到达第一个暂停点 (或者 t=0 时刻的数据)
        # 这里我们直接读取初始状态。根据您的模型，可能需要先 continue 一次，或者直接读。
        # 假设 start 后直接可以读取初始 obs
        obs = self._get_observation()

        return obs

    def step(self, action):
        """
        执行一步：写入动作 -> 继续仿真 -> 等待暂停 -> 读取状态/奖励 -> 判断结束
        """
        # 1. 写入动作
        # 假设 action 是 numpy 数组，Simulink 输入一般需要标量字符串或向量字符串
        # 如果是单动作：
        action_val = float(action[0])
        self.eng.set_param(self.model_name + '/action_input', 'Value', str(action_val), nargout=0)

        # 2. 更新暂停时间并继续
        self.current_pause_time += self.dt
        self.eng.set_param(self.model_name + '/pause_time', 'Value', str(self.current_pause_time), nargout=0)
        self.eng.set_param(self.model_name, 'SimulationCommand', 'continue', nargout=0)

        # 3. 等待 Simulink 暂停 (阻塞式等待)
        # 实际上 MATLAB Engine 的 set_param 是异步的，我们需要轮询状态
        while True:
            status = self.eng.get_param(self.model_name, 'SimulationStatus')
            if status == 'paused':
                break
            if status == 'stopped':
                # 如果仿真意外停止（比如报错或模型内部逻辑导致停止）
                break
            # 极短的 sleep 避免 CPU 占用过高
            time.sleep(0.001)

        # 4. 获取数据
        obs = self._get_observation()
        reward = self._get_reward()

        # 5. 判断 Done
        done = False
        # 时间结束
        sim_time = self.eng.eval('out.time.Data(end)', nargout=1)
        if sim_time >= self.stop_time or status == 'stopped':
            done = True

        # (可选) 如果有其他失败条件（如倒立摆倒下），也可以在这里加：
        # if abs(obs[0]) > limit: done = True

        info = {'time': sim_time}

        return obs, reward, done, info

    def _get_observation(self):
        # 辅助函数：从 workspace 读取 obs
        # 注意：这里假设 Simulink 输出了名为 out.obs 的数据
        try:
            # 获取最后一行数据
            obs = np.array(self.eng.eval('out.obs.Data(end,:)', nargout=1)).flatten()
            return obs.astype(np.float32)
        except Exception as e:
            print(f"读取 Obs 失败: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_reward(self):
        try:
            reward = float(self.eng.eval('out.reward.Data(end)', nargout=1))
            return reward
        except:
            return 0.0

    def close(self):
        self.eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
        self.eng.quit()