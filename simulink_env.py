import random

import gym
from gym import spaces
import numpy as np
import matlab.engine
import os
import time
import math


class SimulinkGymEnv(gym.Env):
    """
    Simulink 模型交互的标准 Gym 环境封装
    """

    def __init__(self, model_name='RLmodel', dt=0.05, stop_time=20.0, debug_mode=False):
        """
        :param debug_mode:
            False (默认) -> 适用于并行训练。每次都启动全新的 MATLAB 进程，互不干扰。
            True  -> 适用于调试。尝试连接已打开的共享 MATLAB 窗口。
        """
        super(SimulinkGymEnv, self).__init__()

        # 1. 配置参数
        self.model_name = model_name
        self.dt = dt
        self.stop_time = stop_time
        self.debug_mode = debug_mode

        # 2. 定义动作空间 (Action Space)
        # 根据你的车辆模型调整 shape=(2,) 和范围
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.state = np.zeros(9, dtype=np.float32)

        # 3. 定义观测空间 (Observation Space)
        # 根据你的 vehicle_dynamics 输出 [s, e, mu, vx, r, beta] 修改 shape=(9, )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        # 【新增】课程学习难度等级
        self.difficulty_level = 0

        # 4. 启动 MATLAB Engine (多进程核心修改部分)
        if debug_mode:
            # --- 调试模式 (单进程) ---
            print(f"[Process {os.getpid()}] 正在调试模式下连接 MATLAB...")
            names = matlab.engine.find_matlab()
            if names:
                print(f"连接到现有共享 MATLAB: {names[0]}")
                self.eng = matlab.engine.connect_matlab(names[0])
            else:
                print("未找到共享 MATLAB，启动新进程并共享...")
                self.eng = matlab.engine.start_matlab()
                self.eng.eval("matlab.engine.shareEngine", nargout=0)
            self.eng.desktop(nargout=0)  # 调试模式下显示界面
        else:
            # --- 训练模式 (多进程并行) ---
            # 必须使用 start_matlab() 启动私有进程，绝不能共享
            print(f"[Process {os.getpid()}] 正在启动独立的 MATLAB Engine (这可能需要几十秒)...")
            self.eng = matlab.engine.start_matlab()
            # 训练模式通常不需要显示界面，节省资源
            # self.eng.desktop(nargout=0)

        # 5. 加载模型
        # 确保每个进程都切换到了正确的目录
        model_path = os.path.abspath('.')
        self.eng.cd(model_path, nargout=0)

        try:
            self.eng.load_system(self.model_name, nargout=0)
            print(f"[Process {os.getpid()}] Simulink 模型 '{self.model_name}' 加载成功。")
        except Exception as e:
            print(f"[Process {os.getpid()}] 模型加载失败: {e}")
            # 如果是并行环境，这里加载失败通常是因为路径不对，或者 MATLAB 没激活

        self.current_pause_time = 0.0

    def set_difficulty(self, level):
        """外部调用此方法修改难度"""
        print(f"[Env] Setting difficulty level to {level}")
        self.difficulty_level = level

    def reset(self):
        """
        重置环境：停止仿真 -> 设置车辆初始状态 -> 启动仿真 -> 返回初始观测
        """
        # ---------------- 1. 停止之前的仿真 ----------------
        try:
            status = self.eng.get_param(self.model_name, 'SimulationStatus')
            if status != 'stopped':
                self.eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
        except:
            pass

        # ---------------- 2. 重置时间控制参数 ----------------
        self.current_pause_time = self.dt

        # 设置仿真总时长 (多给一点冗余时间)
        self.eng.set_param(self.model_name, 'StopTime', str(self.stop_time + self.dt), nargout=0)
        # 重置暂停模块的时间
        self.eng.set_param(self.model_name + '/pause_time', 'Value', str(self.current_pause_time), nargout=0)

        # ---------------- 3. 设置初始动作 (归零) ----------------
        # 车辆模型有两个输入：Fx 和 delta，重置时设为 0
        self.eng.set_param(self.model_name + '/Txr_input', 'Value', '0', nargout=0)
        self.eng.set_param(self.model_name + '/delta_input', 'Value', '0', nargout=0)

        # ---------------- 4. 设置车辆初始状态：根据不同难度等级设置不同的初始状态----------------
        # 车辆状态：[s, e, mu, vx, r, beta, ax, omega_r, vy]
        # === 默认值 ===
        init_s = 0.0  # 起点
        init_e = 0.0  # 位于赛道中心
        init_mu = 0.0
        init_ax = 0.0  # 初始纵向加速度为0
        init_vx = 15.0
        init_r = 0.0
        init_beta = 0.0

        # 根据不同难度设置 vx, r, beta
        if self.difficulty_level == 0:
            # 阶段0：完全直线，仅改变初始速度
            init_vx = init_vx + np.random.uniform(-5.0, 5.0)
            # pass
        elif self.difficulty_level == 1:
            # 阶段1：增加初始位置和航向偏差
            # e: +/- 1m, mu: +/- 0.1 rad
            init_e = np.random.uniform(-1.0, 1.0)
            init_mu = np.random.uniform(-0.1, 0.1)
        elif self.difficulty_level >= 2:
            # 阶段2及以上：增加车辆不稳定状态 (侧滑、横摆)
            # e: +/- 2m, mu: +/- 0.2 rad
            # r: +/- 0.5 rad/s, beta: +/- 0.1 rad
            init_e = np.random.uniform(-2.0, 2.0)
            init_mu = np.random.uniform(-0.2, 0.2)
            init_r = np.random.uniform(-0.5, 0.5)
            init_beta = np.random.uniform(-0.1, 0.1)

        init_omega_r = init_vx / 0.353
        init_vy = init_vx * math.tan(init_beta)

        init_x = [float(init_s), float(init_e), float(init_mu),
                  float(init_vx), float(init_r), float(init_beta),
                  float(init_ax), float(init_omega_r), float(init_vy)]

        # --- 调试模式下在输出端打印初始状态 ---
        if self.debug_mode:
            print(f"🔄 [Episode Start] Level {self.difficulty_level} | "
                  f"vx={init_vx:.2f}, e={init_e:.2f}, mu={init_mu:.3f}, "
                  f"r={init_r:.3f}, beta={init_beta:.3f}")

        # ---------------- 5. 写入 MATLAB 工作区 ----------------
        # 注意：Simulink 模型中的 Integrator (积分) 模块必须设置为读取这些变量
        self.eng.workspace['x0'] = matlab.double(init_x)

        # ---------------- 6. 启动仿真 ----------------
        self.eng.set_param(self.model_name, 'SimulationCommand', 'start', nargout=0)

        # ---------------- 7. 返回初始观测值 ----------------
        # 不需要去 Simulink 读，因为是我们刚设定的，直接构建 array 返回最快且最准
        obs = np.array([init_s, init_e, init_mu, init_vx, init_r,
                        init_beta, init_ax, init_omega_r, init_vy], dtype=np.float32)
        # 同步更新内部状态
        self.state = obs
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        return obs

    def step(self, action):
        """
        执行一步：写入动作 -> 继续仿真 -> 等待暂停 -> 读取状态/奖励 -> 判断结束
        """
        # 1. 写入动作
        # RL 输出通常归一化在 [-1, 1]，需要映射回物理意义
        # 假设：最大驱动力 5000N，最大转向角 0.5 rad (约28度)
        MAX_TRACTION = 1000.0
        MAX_STEER = 30 / 180 * 3.14

        Txr_val = float(action[0]) * MAX_TRACTION
        delta_val = float(action[1]) * MAX_STEER
        # 写入Simulink
        self.eng.set_param(self.model_name + '/Txr_input', 'Value', str(Txr_val), nargout=0)
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
        self.state = obs

        reward, done_logic, info_reward = self._calculate_reward(obs, action, self.last_action)
        self.last_action = action.copy()

        # 5. 判断 Done
        done = False

        if done_logic:
            done = True

        # 时间结束
        try:
            current_sim_time = self.eng.eval('out.time(end)', nargout=1)
        except:
            current_sim_time = self.current_pause_time

        if current_sim_time >= self.stop_time:
            done = True
            info_reward['is_success'] = False  # 超时不算成功

        # Simulink 意外停止
        if status == 'stopped' and not done:
            print('Warning: Simulation stopped unexpectedly.')
            done = True

        info = {'time': current_sim_time}
        info.update(info_reward)

        if done and self.debug_mode:
            full_data, full_time = self._get_final_states()

            if full_data is not None:
                # 将大数据放到 info 字典中
                info['episode_history'] = {
                    'time': full_time,
                    'data':full_data
                }

        return obs, reward, done, info

    def _get_observation(self):
        # 辅助函数：从 workspace 读取 obs
        # 注意：这里假设 Simulink 输出了名为 out.obs 的数据
        try:
            # 获取最后一行数据
            obs = np.array(self.eng.eval('out.obs(:,:,end)', nargout=1)).flatten()
            return obs.astype(np.float32)
        except Exception as e:
            print(f"读取 Obs 失败: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_final_states(self):
        """
        辅助函数：从 workspace 读取完整的历史数据
        """
        try:
            full_data = np.squeeze(np.array(self.eng.eval("out.obs(1,:,:)", nargout=1)))
            full_time = np.array(self.eng.eval("out.time", nargout=1))
            full_time = full_time.flatten()
            print(f" [Env] 成功提取历史数据：{full_data.shape} points")
            return full_data, full_time
        except Exception as e:
            print(f"读取最终 Obs 失败，没有收集到episode结束时的所有状态变量信息: {e}")
            return None, None

    def _calculate_reward(self, obs, current_action, last_action):
        """
        奖励函数设计：最快通过直线赛道
        核心原则：
        1. 奖励前进：沿赛道方向的速度/位移
        2. 惩罚失控：过大的侧滑、横摆
        3. 惩罚出界：软边界+硬边界
        4. 惩罚抖动：动作平滑性
        """
        track_width = 10.0
        track_length = 100.0

        s, e, mu, vx, r, beta, ax, omega_r, vy = obs

        reward = 0.0

        # ============ 1. 前进奖励（最重要）============
        # 奖励沿赛道方向的速度分量
        # Vx是车身纵向速度，Vy是车身横向速度
        # e_phi是车身与赛道方向的夹角
        V_along_track = vx * np.cos(mu) - vy * np.sin(mu)
        # 归一化并给予奖励（假设期望速度范围0-30m/s）
        max_expected_speed = 30.0
        progress_reward = V_along_track / max_expected_speed
        reward += 2.0 * progress_reward
        # 额外奖励：位移增量（可选，与速度奖励二选一或组合使用）
        # delta_s = s - last_s
        # reward += 0.5 * delta_s

        # ============ 2. 稳定性约束 ============
        # 2.1 惩罚过大的质心侧偏角（防止漂移失控）
        beta_threshold = np.radians(12)  # 12度以内不惩罚
        beta_max = np.radians(30)  # 超过30度严重惩罚

        if abs(beta) > beta_threshold:
            beta_excess = abs(beta) - beta_threshold
            beta_range = beta_max - beta_threshold
            beta_penalty = -0.5 * (beta_excess / beta_range) ** 2
            reward += beta_penalty

        # 2.2 惩罚过大的横摆角速度（防止打转）
        r_threshold = np.radians(25)  # 25度/秒以内不惩罚
        r_max = np.radians(60)  # 超过60度/秒严重惩罚

        if abs(r) > r_threshold:
            r_excess = abs(r) - r_threshold
            r_range = r_max - r_threshold
            r_penalty = -0.3 * (r_excess / r_range) ** 2
            reward += r_penalty

        # 2.3 惩罚过大的航向偏差（车头方向偏离赛道太多）
        mu_threshold = np.radians(20)  # 20度以内不惩罚
        mu_max = np.radians(60)

        if abs(mu) > mu_threshold:
            e_phi_excess = abs(mu) - mu_threshold
            e_phi_range = mu_max - mu_threshold
            e_phi_penalty = -0.3 * (e_phi_excess / e_phi_range) ** 2
            reward += e_phi_penalty

        # ============ 3. 边界约束 ============
        track_half_width = track_width / 2.0

        # 3.1 软边界惩罚：接近边界时逐渐增加惩罚
        boundary_margin = 1.0  # 距离边界1m开始警告
        dist_to_boundary = track_half_width - abs(e)

        if boundary_margin > dist_to_boundary > 0:
            # 使用指数型惩罚，越接近边界惩罚增长越快
            boundary_penalty = -0.8 * np.exp(-3 * dist_to_boundary / boundary_margin)
            reward += boundary_penalty

        # 3.2 硬边界惩罚：超出边界（这种情况通常会触发done）
        if abs(e) >= track_half_width:
            reward += -10.0  # 严重惩罚

        # ============ 4. 动作平滑性（抑制抖动）============
        # 需要在__init__中初始化: self.last_action = 0.0
        current_steer = current_action[1]
        current_traction = current_action[0]
        last_steer = last_action[1]
        last_traction = last_action[0]

        steer_change = abs(current_steer - last_steer)
        traction_change = abs(current_traction - last_traction)

        # 归一化动作变化（假设动作范围[-1, 1]，最大变化为2）
        max_action_change = 2.0
        steer_normalized_change = steer_change / max_action_change
        traction_normalized_change = traction_change / max_action_change

        # 平滑性惩罚
        smoothness_penalty = -0.3 * (steer_normalized_change ** 2) - 0.1 * (traction_normalized_change ** 2)
        reward += smoothness_penalty

        # ============ 5. 终点奖励 ============
        # 成功到达终点给予大奖励
        if s >= track_length:
            # 根据用时给予额外奖励（用时越短奖励越高）
            time_bonus = max(0.0, 10.0 - self.current_pause_time * 0.5)
            reward += 20.0 + time_bonus

        # ============ 6. 存活奖励（可选）============
        # 每一步给予小的存活奖励，鼓励agent保持在赛道内
        reward += 0.1

        """
        检查是否结束episode
        """
        done = False
        is_success = False

        # 情况 A: 成功冲线
        if s >= track_length:
            done = True
            is_success = True
            print(f"Success! Reached target at time {self.current_pause_time:.2f}, reward = {reward:.2f}")

        # 情况 B: 冲出赛道 (Fail)
        elif abs(e) > track_half_width:
            done = True
            print(f"Failed! Out of track (e={e:.2f}), reward = {reward:.2f}")

        # 情况 C: 车辆完全失控 (掉头或侧滑过大)
        # mu > 90度 (1.57 rad) 或 beta > 1.0 rad
        elif abs(mu) > 1.57 or abs(beta) > 1.0:
            done = True
            print(f"Failed! Unstable (mu={mu:.2f}, beta={beta:.2f}), reward = {reward:.2f}")

        info = {
                'reward_progress': progress_reward,
                'is_success': is_success
            }

        return reward, done, info

    # def calculate_reward(self, obs, action, last_action):
    #     """
    #     计算奖励函数
    #     输入:
    #         obs: [s, e, mu, vx, r, beta] (须与 Matlab 输出顺序一致)
    #         action: [Fx_norm, delta_norm] (归一化后的动作)
    #     输出:
    #         reward: float
    #         done: bool (逻辑上的结束)
    #         info: dict (调试信息)
    #     """
    #     # 提取状态 (根据你的 vehicle_dynamics 输出顺序)
    #     s, e, mu, vx, r, beta, ax, omega_r, vy = obs
    #
    #     # 动作 (用于计算平滑惩罚)
    #     current_steer = action[1]
    #     last_steer = last_action[1]
    #
    #     # 转向平滑性惩罚
    #     reward_smooth = -10.0 * ((current_steer - last_steer) ** 2)
    #
    #     # --- 1. 进度奖励 (Progress) ---
    #     # 鼓励沿赛道方向的高速行驶，不稳定时对速度的奖励变少？
    #     # 系数 1.0 可以根据 vx 的大小调整，vx~15时 reward~15
    #     reward_progress = 0.5 * (vx * np.cos(mu))
    #
    #     # --- 2. 稳定性惩罚 (Stability) ---
    #     # 关键：beta 和 mu 必须重罚，否则车会横着走
    #     # e 的惩罚系数可以小一点，允许轻微偏离中心
    #     reward_stability = - 0.5 * abs(e) \
    #                        - 5.0 * abs(mu) \
    #                        - 10.0 * abs(beta) \
    #                        - 1.0 * abs(r)
    #
    #     # --- 3. 动作平滑惩罚 (Action Cost) ---
    #     # 惩罚大幅度打方向盘，减少震荡
    #     reward_action = -0.5 * (current_steer ** 2)
    #
    #     # --- 4. 终端判断与奖励 (Terminal) ---
    #     reward_terminal = 0.0
    #     done = False
    #     is_success = False
    #
    #     # 赛道宽 10m => 左右偏差限幅 +/- 5m
    #     TRACK_WIDTH_HALF = 10.0
    #     TRACK_LENGTH = 100.0
    #
    #     # 情况 A: 成功冲线
    #     if s >= TRACK_LENGTH:
    #         reward_terminal = 500.0
    #         done = True
    #         is_success = True
    #         print(f"Success! Reached target at time {self.current_pause_time:.2f}")
    #
    #     # 情况 B: 冲出赛道 (Fail)
    #     elif abs(e) > TRACK_WIDTH_HALF:
    #         reward_terminal = -500.0
    #         done = True
    #         print(f"Failed! Out of track (e={e:.2f})")
    #
    #     # 情况 C: 车辆完全失控 (掉头或侧滑过大)
    #     # mu > 90度 (1.57 rad) 或 beta > 1.0 rad
    #     elif abs(mu) > 1.57 or abs(beta) > 1.0:
    #         reward_terminal = -500.0
    #         done = True
    #         print(f"Failed! Unstable (mu={mu:.2f}, beta={beta:.2f})")
    #
    #     # 总分
    #     total_reward = reward_progress + reward_stability + reward_action + reward_terminal
    #
    #     # 缩放总分 (可选，为了让数值在 -10 到 10 之间，利于神经网络收敛)
    #     total_reward = total_reward * 0.1
    #
    #     info = {
    #         'reward_progress': reward_progress,
    #         'reward_stability': reward_stability,
    #         'is_success': is_success
    #     }
    #     return total_reward, done, info

    def close(self):
        self.eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
        self.eng.quit()