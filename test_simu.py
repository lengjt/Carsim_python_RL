import matlab.engine
import numpy as np
import time
import os


# ==========================================
# 模拟一个简单的 Agent (替代 SAC)
# ==========================================
class RandomAgent:
    def __init__(self, action_dim, action_bound):
        self.action_dim = action_dim
        self.action_bound = action_bound

    def get_action(self, obs):
        # 这里仅为了测试流程，输出随机动作
        # 实际训练时，这里替换为您 SAC 的 policy_net.get_action(obs)
        action = np.random.uniform(-1, 1, size=(self.action_dim,))
        return action


# ==========================================
# 主程序
# ==========================================
def main():
    print("正在启动 MATLAB Engine，请稍候...")
    # 连接到已有的 MATLAB 窗口，无需等待启动，直接就能交互
    names = matlab.engine.find_matlab()
    if names:
        eng = matlab.engine.connect_matlab(names[0])
    else:
        eng = matlab.engine.start_matlab()  # 如果没找到再启动新的
    print("MATLAB Engine 启动成功！")
    eng.desktop(nargout=0)

    # === 配置参数 ===
    env_name = 'rlSimplePendulumModel'
    model_path = os.path.abspath('.')  # 假设模型在当前目录
    eng.cd(model_path, nargout=0)

    try:
        # eng.load_system(env_name)
        eng.open_system(env_name, nargout=0)
        print(f"Simulink 模型 '{env_name}' 加载成功。")
    except Exception as e:
        print(f"错误: 无法加载模型 '{env_name}'。请确认文件名正确且在当前路径下。\n{e}")
        return

    # 定义仿真参数
    sample_time = 0.05  # Python 介入的周期 (s)
    stop_time = 10.0  # 单个 Episode 总时长 (s)
    max_episodes = 3  # 测试跑 3 个回合

    action_dim = 1
    action_bound = 2.0  # 倒立摆力矩限制

    agent = RandomAgent(action_dim, action_bound)

    print(f"\n开始测试流程: 共 {max_episodes} 个回合, 步长 {sample_time}s")

    for ep in range(max_episodes):
        print(f"\n=== Episode {ep + 1} Start ===")

        # 1. 重置环境参数
        # 设置总仿真时间 (多给一点点冗余，防止刚好在结束点无法暂停)
        eng.set_param(env_name, 'StopTime', str(stop_time + sample_time), nargout=0)
        # 初始化暂停时间点 (第一次暂停的时间)
        current_pause_time = sample_time
        eng.set_param(env_name + '/pause_time', 'value', str(current_pause_time), nargout=0)
        # 初始化控制输入 (假设输入模块名叫 'action_input')
        eng.set_param(env_name + '/action_input', 'value', '0', nargout=0)

        # 2. 开始仿真
        eng.set_param(env_name, 'SimulationCommand', 'start', nargout=0)

        done = False
        step = 0
        episode_reward = 0

        # 3. Step 循环
        while not done:
            # 获取仿真状态: 'stopped', 'initializing', 'running', 'paused'
            model_status = eng.get_param(env_name, 'SimulationStatus')

            if model_status == 'paused':
                # --- A. 从 Simulink 获取数据 (Observation, Reward, Time) ---
                # 注意: 这里的 'logsout' 取决于您 Simulink "To Workspace" 模块的设置
                # 建议在 Simulink 设置中开启 "Single simulation output" 并命名为 "out"

                try:
                    # 获取最新时刻的数据
                    # 假设 Simulink 输出的结构体叫 out，里面有 obs, reward, time
                    # Data[-1] 取最后一行（最新时刻）
                    sim_time = eng.eval('out.time.Data(end)', nargout=1)
                    obs = np.array(eng.eval('out.obs.Data(end,:)', nargout=1)).flatten()
                    reward = float(eng.eval('out.reward.Data(end)', nargout=1))

                    # 打印部分 log 证明拿到了数据
                    if step % 10 == 0:
                        print(f"Step {step:03d} | Time: {sim_time:.2f}s | Obs: {obs} | Reward: {reward:.4f}")

                except Exception as e:
                    print(f"数据读取错误: {e}")
                    break

                # --- B. 产生动作 (Action) ---
                raw_action = agent.get_action(obs)
                # 缩放动作
                real_action = raw_action * action_bound
                real_action = np.clip(real_action, -action_bound, action_bound)

                # --- C. 判断是否结束 (Done) ---
                # 如果时间超过设定，或者触发了某些终止条件(例如倒立摆倒下)
                if sim_time >= stop_time:
                    done = True
                    eng.set_param(env_name, 'SimulationCommand', 'stop', nargout=0)
                    print(f"Episode {ep + 1} 完成。总 Reward: {episode_reward:.4f}")
                    break

                # --- D. 将动作写入 Simulink 并继续 ---
                # 更新下一次暂停的时间
                current_pause_time += sample_time

                # 写入参数
                eng.set_param(env_name + '/action_input', 'Value', str(real_action[0]), nargout=0)
                eng.set_param(env_name + '/pause_time', 'Value', str(current_pause_time), nargout=0)

                # 继续仿真
                eng.set_param(env_name, 'SimulationCommand', 'continue', nargout=0)

                episode_reward += reward
                step += 1

            elif model_status == 'stopped':
                # 如果仿真异常停止
                print("Simulink 意外停止 (可能触发了内部错误或完成)。")
                break

            elif model_status == 'running':
                # 如果正在跑，就等待一下，防止 CPU 空转太快
                # time.sleep(0.001)
                pass
    print("\n测试结束")
    input("按回车键关闭MATLAB并退出Python程序")

    # eng.quit()


if __name__ == '__main__':
    main()