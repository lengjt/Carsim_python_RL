import os
import time
import numpy as np
from stable_baselines3 import SAC
from simulink_env import SimulinkGymEnv


def main():
    # --- 配置 ---
    model_path = "./logs/sac_model_10000_steps.zip"  # 这里指定你想测试的模型（是最优的？还是最新的？）
    model_name_slx = 'rlSimplePendulumModel'

    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return

    print(f"正在加载模型: {model_path}")
    # 加载模型，不需要传入 env，因为预测时不需要 env 的训练逻辑
    model = SAC.load(model_path)

    print("启动 Simulink 环境进行测试...")
    # 注意：这里我们手动实例化环境，方便控制 Render 和窗口
    env = SimulinkGymEnv(model_name_slx, dt=0.05, stop_time=20.0)

    # 强制打开 Simulink 窗口以便观察 (如果 env 内部没写，这里补充)
    # env.eng.desktop(nargout=0)
    # env.eng.open_system(model_name_slx, nargout=0)

    num_test_episodes = 3

    for ep in range(num_test_episodes):
        print(f"=== 测试回合 {ep + 1} ===")
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            # deterministic=True 表示使用确定性策略（不加随机噪声），这是测试时的标准做法
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

            # 可以在这里打印每一步的信息
            # print(f"Step {step}: Action={action}, Reward={reward:.4f}")

        print(f"回合结束。总步数: {step}, 总奖励: {total_reward:.4f}")
        time.sleep(1)  # 休息一下方便观察 Simulink 窗口

    print("所有测试完成。")
    env.close()


if __name__ == '__main__':
    main()