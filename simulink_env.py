import random

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

    def __init__(self, model_name='Pendulum_Model', dt=0.05, stop_time=20.0, new_process=False):
        super(SimulinkGymEnv, self).__init__()

        # 1. 配置参数
        self.model_name = model_name
        self.dt = dt
        self.stop_time = stop_time

        # 2. 定义动作空间 (Action Space)
        # 例如：倒立摆是 1 维控制 (力矩)，范围 [-2, 2]
        # 如果是车辆模型，请修改 shape=(2,) 并调整 low/high
        self.action_space = spaces.Box(low=[-5000.0, -35/180*3.14], high=[5000.0, 35/180*3.14], shape=(2,), dtype=np.float32)

        # 3. 定义观测空间 (Observation Space)
        # 例如：[cos(theta), sin(theta), dot_theta] 维度为 3
        # 请根据您 Simulink 输出的 obs 维度修改 shape
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        if new_process:
            print("为评估环境启动独立的MATLAB进程...")
            self.eng = matlab.engine.start_matlab()
        else:
            # 4. 启动 MATLAB Engine
            print("正在启动 MATLAB Engine，请稍候...")
            # 连接到已有的 MATLAB 窗口，无需等待启动，直接就能交互
            names = matlab.engine.find_matlab()
            if names:
                print(f"连接到现有共享MATLAB：{names[0]}")
                self.eng = matlab.engine.connect_matlab(names[0])
            else:
                print("启动新的共享MATLAB进程")
                self.eng = matlab.engine.start_matlab()
                self.eng.eval("matlab.engine.shareEngine", nargout=0)
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

        # 随机数设置状态初始值
        rand_theta = np.random.uniform(-np.pi, np.pi)
        rand_theta_dot = np.random.uniform(-1.0, 1.0)
        self.eng.workspace['theta0'] = float(rand_theta)
        self.eng.workspace['theta_dot0'] = float(rand_theta_dot)

        # 启动仿真
        self.eng.set_param(self.model_name, 'SimulationCommand', 'start', nargout=0)

        # 等待到达第一个暂停点 (或者 t=0 时刻的数据)
        # 这里我们直接读取初始状态。根据您的模型，可能需要先 continue 一次，或者直接读。
        # 假设 start 后直接可以读取初始 obs
        obs = np.array([rand_theta, rand_theta_dot], dtype=np.float32)

        return obs

    def step(self, action):
        """
        执行一步：写入动作 -> 继续仿真 -> 等待暂停 -> 读取状态/奖励 -> 判断结束
        """
        # 1. 写入动作
        # 假设 action 是 numpy 数组，Simulink 输入一般需要标量字符串或向量字符串
        # 如果是单动作：
        Fx_val = float(action[0])
        delta_val = float(action[1])
        self.eng.set_param(self.model_name + '/Fx_input', 'Value', str(Fx_val), nargout=0)
        self.eng.set_param(self.model_name + '/delta_input', 'Value', str(delta_val), nargout=0)

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
        reward, done_simu = self.calculate_reward(obs, action)

        # 5. 判断 Done
        done = False
        # 时间结束
        sim_time = self.eng.eval('out.time.Data(end)', nargout=1)
        if sim_time >= self.stop_time or status == 'stopped':
            done = True
        # simulink因为某种原因返回了done = 1.0
        if done_simu > 0.5:
            print("simulation stopped before reaching simulation time")
            done = True

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

    def calculate_reward(self, obs, action):
        # 提取状态 (注意要和 Matlab 输出顺序一致)
        s, e, mu, vx, r, beta = obs

        # --- 1. 进度奖励 (Progress Reward) ---
        # 鼓励向前移动，且速度越快奖励越高
        # 使用沿赛道方向的速度投影：vx * cos(mu)
        reward_progress = 1.0 * (vx * np.cos(mu))

        # --- 2. 稳定性惩罚 (Stability Penalty) ---
        # 核心：必须惩罚大的侧偏角和横摆角速度，否则车辆会失控
        # e: 偏离中心越远，惩罚越大
        # mu: 车头不正，惩罚
        # beta: 侧滑严重，重罚 (这是你任务的关键)
        reward_stability = - 2.0 * abs(mu) \
                           - 5.0 * abs(beta) \
                           - 0.5 * abs(r)

        # --- 3. 动作平滑惩罚 (Control Effort) ---
        # 避免方向盘高频抖动
        # assuming action[1] is steering angle delta
        reward_action = -0.1 * (action[2] ** 2)

        # --- 4. 终端奖励/惩罚 (Terminal conditions) ---
        reward_terminal = -1

        # 情况A: 冲出赛道 (Fail)
        # 赛道宽10m -> e范围 [-5, 5]
        if abs(e) > 5.0:
            reward_terminal = -1000.0
            done = True

        # 情况B: 完成比赛 (Success)
        elif s >= 200.0:
            reward_terminal = +500.0
            # 额外的时间奖励：步数越少分越高(通常RL框架自带gamma衰减，这里可以给个大额固定分)
            done = True

        # 情况C: 车辆失控 (Optional)
        # 如果车头完全调转 (mu > 90度) 或者侧滑角过大
        elif abs(mu) > np.pi / 2 or abs(beta) > 1.0:
            reward_terminal = -1000.0
            done = True
        else:
            done = False

        # 总 Reward
        total_reward = reward_progress + reward_stability + reward_action + reward_terminal

        # 归一化 (可选，建议将 reward 控制在 [-1, 1] 或 [-10, 10] 区间以便训练)
        return total_reward, done

    def _get_done(self):
        try:
            done = float(self.eng.eval('out.done.Data(end)', nargout=1))
            return done
        except:
            return 0.0

    def close(self):
        self.eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
        self.eng.quit()