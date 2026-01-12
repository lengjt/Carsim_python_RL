import gym
import os
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from simulink_env import SimulinkGymEnv # 你的环境文件


def main():

    # --- 1. 路径设置 ---
    log_dir = "./logs_curriculum/"
    model_path = os.path.join(log_dir, "model_stage_0_level0.zip") # 或者 latest_model.zip
    stats_path = os.path.join(log_dir, "model_stage_0_level0_vecnormalize.pkl") # 必须要对应的 pkl 文件！

    # --- 2. 创建一模一样的环境结构 ---
    # 必须和训练时保持一致的 Wrappers
    env = DummyVecEnv([lambda: SimulinkGymEnv(model_name='RLmodel', debug_mode=True)])
    env = VecFrameStack(env, n_stack=4)

    # --- 3. 加载归一化参数 (关键步骤！) ---
    # 这一步如果不做，模型表现会非常差，因为输入数据的分布对不上
    env = VecNormalize.load(stats_path, env)
    # 测试时不要更新均值方差，也不要归一化 Reward
    env.training = False
    env.norm_reward = False

    # --- 设置测试过程的难度 ---
    level = 0
    env.env_method('set_difficulty', level)

    # --- 4. 加载模型 ---
    model = SAC.load(model_path)

    # --- 5. 享受成果 ---
    print("开始测试模型...")
    for ep in range(1):
        obs = env.reset()
        print(f" Eval Obs (Norm): {obs}")
        done = False
        total_reward = 0
        while not done:
            # deterministic=True 关闭随机探索，展示最优策略
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, infos = env.step(action)
            total_reward += reward

            # 检查 info 是否包含我们需要的数据
            if 'episode_history' in infos[0]:
                print(" 检测到完整历史数据，准备绘图... ")
                history_data = infos[0]['episode_history']
                plot_simulink_data(history_data)
                break
        print(f"Episode {ep+1}: Reward = {total_reward}")
    env.close()


def plot_simulink_data(history):
    """
    绘制 Simulink 原生数据
    history: {'time': array, 'data': array (N, 6)}
    """
    t = history['time']
    data = history['data']

    # 假设变量顺序是: [s, e, mu, vx, r, beta]
    # 请根据你的 Simulink Outport 实际顺序修改索引
    s = data[0, :]
    e = data[1, :]
    mu = data[2, :]
    vx = data[3, :]
    r = data[4, :]
    beta = data[5, :]
    ax = data[6, :]
    omega_r = data[7, :]
    vy = data[8, :]

    # 开始画图
    plt.style.use('seaborn-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(5, 6), sharex=True)

    # 1. 速度 vx
    axes[0].plot(t, vx, label='Velocity ($v_x$)', color='blue')
    axes[0].axhline(15, color='red', linestyle='--', alpha=0.5, label='Target')
    axes[0].set_ylabel('Speed (m/s)')
    axes[0].set_title('Velocity Tracking (Simulink Raw Data)')
    axes[0].legend()
    axes[0].grid(True)

    # 2. 稳定性 (r, beta)
    ax2_right = axes[1].twinx()
    axes[1].plot(t, r * 57.3, label='Yaw Rate ($r$)', color='orange')  # 转为 deg
    ax2_right.plot(t, beta * 57.3, label='Side Slip ($\\beta$)', color='green', linestyle='--')

    axes[1].set_ylabel('Yaw Rate (deg/s)')
    ax2_right.set_ylabel('Side Slip (deg)')

    # 合并图例
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    axes[1].set_title('Stability States')

    # 3. 轨迹误差 (s, e)
    # 这里画 e 随时间的变化，或者 e 随 s 的变化
    axes[2].plot(t, e, label='Lateral Error ($e$)', color='purple')
    axes[2].plot(t, mu * 57.3, label='Heading Error ($\\mu$ deg)', color='brown', alpha=0.6)
    axes[2].set_ylabel('Error')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend()
    axes[2].set_title('Tracking Errors')

    plt.tight_layout()
    plt.show()
    print("✅ 绘图完成")


if __name__ == '__main__':
    main()
