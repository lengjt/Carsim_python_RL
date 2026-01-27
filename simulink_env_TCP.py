import gym
from gym import spaces
import numpy as np
import matlab.engine
import os
import time
import math
import socket
import struct  # 用于打包/解包二进制数据

class SimulinkGymEnv(gym.Env):
    """
    Simulink 模型交互的标准 Gym 环境封装 (TCP/IP 高速版)
    """

    def __init__(self, model_name='RLmodel', ctrl_dt=0.01, ts=0.001, stop_time=20.0, debug_mode=False):

        # 1. 配置参数
        self.model_name = model_name
        self.ts = ts
        self.ctrl_dt = ctrl_dt   # 控制周期
        self.stop_time = stop_time
        self.debug_mode = debug_mode
        
        # TCP 配置
        self.host = '127.0.0.1'
        self.port = 54321  # 确保和 Simulink 里的 TCP Send/Receive 端口一致
        self.conn = None
        self.addr = None

        # 2. 定义动作空间 (Action Space)
        # 动作: 方向盘转角增益 [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.last_action = np.array([0.0], dtype=np.float32)

        # 3. 定义观测空间 (Observation Space)
        # 假设 Simulink 发送的数据包含: Obs(6) + Mu(4) = 10个 double
        # 如果你只发 6 个，请修改这里的 buffer size 计算逻辑
        self.obs_dim = 6
        self.mu_dim = 4 
        self.total_recv_dim = self.obs_dim + self.mu_dim 
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # 4. 启动 TCP Server (Python 作为服务端)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"🚀 TCP Server 启动，监听端口 {self.port}...")

        # 5. 启动 MATLAB Engine (保留用于管理仿真启停)
        if debug_mode:
            print(f"[Process {os.getpid()}] 正在调试模式下连接 MATLAB...")
            names = matlab.engine.find_matlab()
            if names:
                print(f"连接到现有共享 MATLAB: {names[0]}")
                self.eng = matlab.engine.connect_matlab(names[0])
            else:
                print("未找到共享 MATLAB，启动新进程并共享...")
                self.eng = matlab.engine.start_matlab()
                self.eng.eval("matlab.engine.shareEngine", nargout=0)
            self.eng.desktop(nargout=0)
        else:
            print(f"[Process {os.getpid()}] 正在启动独立的 MATLAB Engine...")
            self.eng = matlab.engine.start_matlab()

        # 6. 加载模型
        model_path = os.path.abspath('.')
        self.eng.cd(model_path, nargout=0)
        try:
            self.eng.load_system(self.model_name, nargout=0)
            print(f"[Process {os.getpid()}] Simulink 模型 '{self.model_name}' 加载成功。")
        except Exception as e:
            print(f"[Process {os.getpid()}] 模型加载失败: {e}")

        self.current_pause_time = 0.0

    def reset(self):
        """
        重置环境：停止仿真 -> 启动仿真 -> 等待 TCP 连接 -> 接收初始观测
        """
        # 1. 停止之前的仿真 (通过 Engine)
        try:
            status = self.eng.get_param(self.model_name, 'SimulationStatus')
            if status != 'stopped':
                self.eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
            
            # 关闭旧连接
            if self.conn:
                self.conn.close()
                self.conn = None
        except:
            pass

        # 2. 设置仿真参数 (通过 Engine)
        # 注意：不再需要 pause_time 参数，因为 TCP Receive 会自动阻塞 Simulink
        self.eng.set_param(self.model_name, 'StopTime', str(self.stop_time), nargout=0)

        # 设置通讯速率 -- ctrl_dt
        self.eng.workspace['ctrl_dt'] = float(self.ctrl_dt)
        
        # 3. 启动仿真
        # Simulink 启动后，模型里的 TCP Send 会尝试连接 Python
        self.eng.set_param(self.model_name, 'SimulationCommand', 'start', nargout=0)

        # 4. 阻塞等待 Simulink 连接
        print("⏳ 等待 Simulink 建立 TCP 连接...")
        self.conn, self.addr = self.server_socket.accept()
        print(f"✅ Simulink 已连接: {self.addr}")

        # 5. 接收初始观测值 (S0)
        # Simulink 在第一步会先发状态，再收动作
        obs, _ = self._recv_tcp_data()
        
        self.steps = 0
        return obs

    def step(self, action):
        """
        执行一步：发送动作(TCP) -> 接收状态(TCP) -> 计算奖励
        """
        # 1. 处理动作数据
        MAX_STR_ADD = 90/180*3.14
        Steer_add = float(action[0]) * MAX_STR_ADD
        
        # 2. 发送动作到 Simulink (TCP Send)
        # 打包为 double (8字节)，假设 Simulink TCP Receive 设置为 double, size=1
        try:
            # '<d' 表示小端序 double
            msg = struct.pack('<d', Steer_add) 
            self.conn.sendall(msg)
        except (BrokenPipeError, ConnectionResetError):
            print("❌ 连接断开 (发送失败)")
            return np.zeros(6), 0, True, {}

        # 3. 接收下一帧状态 (TCP Recv - 阻塞等待)
        # 这一步替代了原来的 while 循环和 pause 逻辑
        obs, mu_tire = self._recv_tcp_data()
        
        # 如果返回 None，说明 Simulink 停止了
        if obs is None:
            done = True
            # 尝试通过 Engine 获取最后的数据用于 debug
            if self.debug_mode:
                 # 这里依然可以用 Engine 读取 workspace 的历史数据
                 # 前提是 Simulink 里 To Workspace 模块还在工作
                pass 
            return np.zeros(6), 0, True, {}

        # 4. 计算奖励
        # 注意：这里需要根据你实际的 obs 结构调整
        reward = self._calculate_reward(obs, action, mu_tire)
        
        # 5. 判断结束
        done = False
        self.steps += 1
        
        # 简单的时间判定 (基于步数估算，或者在 obs 里包含 sim_time)
        current_time = self.steps * self.ctrl_dt
        if current_time >= self.stop_time:
            done = True

        # 如果需要获取完整历史数据用于绘图，可以在 done 时读取 Workspace
        info = {'time': current_time}
        
        if done and self.debug_mode:
            full_data = self._get_final_states()
            if full_data is not None:
                info['episode_history'] = {'data': full_data}

        return obs, reward, done, info

    def _recv_tcp_data(self):
        """
        底层函数：从 TCP 接收并解包数据
        """
        try:
            # 计算需要接收的字节数: (Obs维数 + Mu维数) * 8字节
            # 假设 Simulink 把 Obs(6) 和 Mu(4) Mux 在一起发送
            recv_bytes = 8 * self.total_recv_dim
            
            data = self.conn.recv(recv_bytes)
            
            if not data or len(data) != recv_bytes:
                # 仿真结束
                return None, None
            
            # 解包: 小端序, 10个 double
            all_data = np.array(struct.unpack(f'<{self.total_recv_dim}d', data))
            
            # 拆分 Obs 和 Mu
            obs = all_data[:self.obs_dim].astype(np.float32)
            mu = all_data[self.obs_dim:].astype(np.float32)
            
            return obs, mu
            
        except Exception as e:
            # print(f"接收数据异常: {e}")
            return None, None

    def _calculate_reward(self, obs, current_action, mu_tire):
        # 你的奖励函数逻辑...
        reward = 0.1
        return reward

    def _get_final_states(self):
        """
        保留原有的 Engine 读取方式，用于 Episode 结束后的数据收集
        """
        try:
            if self.eng.eval("exist('out', 'var')") == 0:
                return None
            full_data = np.array(self.eng.eval("out.obs(:,:)", nargout=1))
            return full_data
        except:
            return None

    def close(self):
        if self.conn:
            self.conn.close()
        if self.server_socket:
            self.server_socket.close()
        try:
            self.eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
            self.eng.quit()
        except:
            pass


def main():
    # --- 配置 ---
    model_name_slx = 'RLmodel'

    print("启动 Simulink 环境进行测试...")
    # 注意：这里我们手动实例化环境，方便控制 Render 和窗口
    env = SimulinkGymEnv(model_name_slx, ctrl_dt=0.01, ts=0.0005, stop_time=30.0, debug_mode=True)

    num_test_episodes = 1

    for ep in range(num_test_episodes):
        print(f"=== 测试回合 {ep + 1} ===")
        obs = env.reset()
        
        # 打印初始观测值
        print(f"初始观测值: X_DM:{obs[0]:.4f}, Y_DM:{obs[1]:.4f}, Str_Car:{obs[2]:.4f},\
               Vx:{obs[3]:.4f}, beta:{obs[4]:.4f}, dYaw:{obs[5]:.4f}")

        done = False
        total_reward = 0
        step = 0
        action_value = 0.0  

        while not done:
            # deterministic=True 表示使用确定性策略（不加随机噪声），这是测试时的标准做法
            action = [action_value]  # 这里使用零动作进行测试，替换为你的策略输出
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
            action_value += 0.01  # 示例：每步增加一点动作值

            # 可以在这里打印每一步的信息
            if step % 10 == 0:
                print(f"Step {step}: Action={action}, Reward={reward:.4f}")

        print(f"回合结束。总步数: {step}, 总奖励: {total_reward:.4f}")
        time.sleep(1)  # 休息一下方便观察 Simulink 窗口

    print("所有测试完成。")
    env.close()


if __name__ == '__main__':
    main()