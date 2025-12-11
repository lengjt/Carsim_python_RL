import gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from simulink_env import SimulinkGymEnv  # 导入刚才定义的类


def main():
    # 1. 实例化环境
    # 请确保 model_name 与您的 slx 文件名一致
    env = SimulinkGymEnv(model_name='rlSimplePendulumModel', dt=0.05, stop_time=20.0)

    # 2. 定义 SAC 模型
    # MlpPolicy: 使用全连接神经网络
    # verbose=1: 打印训练日志
    model = SAC("MlpPolicy", env, verbose=1,
                learning_rate=3e-4,
                batch_size=256,
                buffer_size=50000,  # 如果内存不够可以调小
                tensorboard_log="./sac_simulink_tensorboard/")

    # 3. 设置模型保存回调 (每 1000 步保存一次)
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                             name_prefix='sac_model')

    print("开始训练...")
    # 4. 开始训练
    # total_timesteps: 总交互步数
    model.learn(total_timesteps=10000, callback=checkpoint_callback)

    # 5. 保存最终模型
    model.save("sac_simulink_final")
    print("训练结束，模型已保存。")

    # --- 简单的测试验证 ---
    print("正在演示训练后的模型...")
    obs = env.reset()
    for i in range(200):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # 如果您在 SimulinkEnv 里开启了 desktop=True，这里就能看到 Simulink 里的 Scope 动起来
        if done:
            obs = env.reset()

    env.close()


if __name__ == '__main__':
    main()