import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack, VecMonitor, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
import os

from simulink_env import SimulinkGymEnv


def get_pid_action(raw_obs):
    """
    根据物理观测计算 PID 动作
    Raw Obs: [s, e, mu, vx, r, beta]
    """
    s, e, mu, vx, r, beta, ax, omega_r, vy = raw_obs

    # PID 参数 (需要根据之前测试好的参数填入)
    k_mu, k_r, k_beta = 0.7, 0.2, 0.2

    # 转向控制
    target_steer = - (k_mu * mu) - (k_r * r) - (k_beta * beta)
    steer_action = np.clip(target_steer / 0.5, -1.0, 1.0)  # 假设最大转向0.5

    # --- 2. 油门控制 (状态机) ---
    # 如果车辆非常不稳定，减速；否则加速
    if abs(beta) > 0.2 or abs(r) > 0.2:
        # 失控救车状态：不给油，或者轻微刹车
        throttle_action = 0
    elif abs(beta) < 0.05 or abs(r) < 0.05:
        # 稳定状态：地板油
        throttle_action = 1.0
    else:
        # 中间态：逐渐增加油门
        throttle_action = 6.7 * (abs(r) - 0.05)

    return np.array([throttle_action, steer_action], dtype=np.float32)


def inject_expert_data(env, model, n_steps=1000):
    """
    间断注入专家数据
    env: 已经被 VecNormalize 和 VecFrameStack 包裹的 VecEnv
    """
    print(f"💉 正在注入 PID 专家数据 ({n_steps} steps)...")

    obs = env.reset()
    total_steps = 0

    while total_steps < n_steps:
        # 1. 获取物理观测值 (用于 PID 计算), 获取归一化前的观测值
        raw_states = env.get_attr('state')

        # 2. 计算专家动作 (对每个环境分别计算)
        actions = []
        for i in range(env.num_envs):
            phys_obs = raw_states[i]
            act = get_pid_action(phys_obs)
            actions.append(act)
        actions = np.array(actions)

        # 3. 执行动作
        next_obs, rewards, dones, infos = env.step(actions)

        # 4. 存入 Buffer (存的是处理过的 obs，这就对上了！)
        # 这一步是关键：网络训练用的是 FrameStack+Normalize 后的数据
        model.replay_buffer.add(obs, next_obs, actions, rewards, dones, infos)

        obs = next_obs
        total_steps += env.num_envs

    print(f"💉 注入完成。Buffer Size: {model.replay_buffer.pos}")


def main():
    # --- 配置 ---
    model_name = 'RLmodel'
    log_dir = "./logs_curriculum/"
    os.makedirs(log_dir, exist_ok=True)

    N_ENVS = 2
    # 定义课程阶段
    # level: 难度等级
    # total_timesteps: 该阶段训练总步数
    # eval_threshold: 进入下一阶段所需的平均分 (如果没达到，可能需要重复训练或人工干预)
    STAGES = [
        {'level': 0, 'steps': 50000, 'threshold': 150, 'inject_prob': 0.5},  # 初始阶段，多注入专家数据
        {'level': 1, 'steps': 100000, 'threshold': 120, 'inject_prob': 0.2},  # 中级，少量注入
        {'level': 2, 'steps': 200000, 'threshold': 100, 'inject_prob': 0.05}  # 高级，主要靠自己悟
    ]

    # --- 1. 创建环境 (带 Stack 和 Normalize) ---
    env_kwargs = {'model_name': model_name, 'dt': 0.05, 'stop_time': 20.0, 'debug_mode': False}

    # A. 基础并行环境
    env = make_vec_env(lambda: SimulinkGymEnv(**env_kwargs), n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    # B. FrameStack (感知导数信息)
    # n_stack=4: 能够感知到 位置->速度->加速度->加加速度 的变化趋势
    env = VecFrameStack(env, n_stack=4)

    # C. Normalize (归一化输入和奖励)
    # 这对神经网络收敛至关重要
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # D. Monitor
    env = VecMonitor(env, log_dir)

    # --- 评估环境 (Eval Env) ---
    # 定义评估环境参数 (开启 debug_mode 以便看界面)
    eval_env_kwargs = {'model_name': 'RLmodel', 'dt': 0.05, 'stop_time': 20.0, 'debug_mode': True}

    # 1. 定义一个制造环境的函数 (这是 DummyVecEnv 要求的格式)
    def make_eval_env():
        e = SimulinkGymEnv(**eval_env_kwargs)
        return e

    # 2. 【关键修正】使用 DummyVecEnv 将其转换为 VecEnv
    # DummyVecEnv 适用于单进程 (评估时用这个最好，不会有多进程通信开销)
    eval_env = DummyVecEnv([make_eval_env])

    # 3. 现在它是一个 VecEnv 了，可以安全地添加 VecWrappers
    # 注意：Wrappers 的顺序必须和训练环境完全一致！
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # 4. Normalize
    # 评估时 norm_reward=False (我们要看真实奖励)，training=False (不更新均值方差)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    eval_env = VecMonitor(eval_env, log_dir)

    # --- 2. 初始化模型 ---
    # 检查是否有 Checkpoint
    last_model_path = os.path.join(log_dir, "latest_model.zip")
    last_buffer_path = os.path.join(log_dir, "latest_buffer.pkl")
    if os.path.exists(last_model_path):
        print("加载之前的模型和归一化参数...")
        model = SAC.load(last_model_path, env=env)
        # 加载 Replay Buffer
        if os.path.exists(last_buffer_path):
            print(f"正在加载 Replay Buffer (文件较大，请稍候)...")
            model.load_replay_buffer(last_buffer_path)
            print(f"   Buffer 加载完成，当前样本数: {model.replay_buffer.pos}")
        else:
            print("警告：未找到 Buffer 文件！训练将从空 Buffer 开始（效率较低）。")
        # 加载 VecNormalize 的统计数据 (非常重要！)
        env = VecNormalize.load(os.path.join(log_dir, "latest_vecnormalize.pkl"), env.venv)
    else:
        print("✨ 创建新模型...")
        # 稍微调大 batch_size 和 buffer_size 适应 Stack 后的高维数据
        model = SAC("MlpPolicy", env, verbose=1, batch_size=512, buffer_size=100000, learning_rate=3e-4)

    # --- 3. 课程学习循环 ---
    total_steps_so_far = 0

    for stage_idx, stage in enumerate(STAGES):
        level = stage['level']
        target_steps = stage['steps']
        threshold = stage['threshold']
        inject_prob = stage['inject_prob']

        print(f"\n\n{'=' * 30}")
        print(f"🚀 进入课程阶段 {stage_idx} (难度 Level {level})")
        print(f"目标步数: {target_steps}, 晋级分数: {threshold}")
        print(f"{'=' * 30}\n")

        # 3.1 设置环境难度
        # 由于是 SubprocVecEnv，需要用 env_method 广播
        # 注意：env 现在是包裹了多层的，env_method 会自动穿透到最底层的 GymEnv
        env.env_method('set_difficulty', level)
        # 评估环境也要设
        eval_env.env_method('set_difficulty', level)

        # 3.2 阶段内训练循环
        # 将目标步数拆解为小块，每块之间进行 注入 和 评估
        chunk_size = 5000
        steps_trained = 0

        while steps_trained < target_steps:

            # --- A. 专家数据注入 ---
            # 根据概率决定这轮是否注入
            if np.random.rand() < inject_prob:
                inject_expert_data(env, model, n_steps=1000)

            # --- B. 训练一小段 ---
            model.learn(total_timesteps=chunk_size, reset_num_timesteps=False)
            steps_trained += chunk_size
            total_steps_so_far += chunk_size

            # --- D. 评估 (Test) ---
            if steps_trained % 10000 == 0:  # 每训练1万步测一次
                print("🧐 正在评估当前阶段性能...")
                # 同步归一化参数：把训练环境学到的 mean/std 复制给评估环境
                # 这一步非常关键！否则评估环境不知道怎么归一化，输入全是错的
                eval_env.obs_rms = env.obs_rms

                # 跑 5 个 episode
                mean_reward = 0
                for _ in range(5):
                    obs_e = eval_env.reset()
                    done_e = False
                    ep_reward = 0
                    while not done_e:
                        action_e, _ = model.predict(obs_e, deterministic=True)
                        obs_e, reward_e, done_e, _ = eval_env.step(action_e)
                        ep_reward += reward_e
                    mean_reward += ep_reward
                mean_reward /= 5.0
                print(f" 阶段 {stage_idx} 当前平均分: {mean_reward.item():.2f} (目标: {threshold})")

                # 如果提前达到目标，是否跳过剩余步数？
                # 建议不要跳过，巩固一下比较好。或者设置一个更高的 "Early Pass" 分数。
                if mean_reward > threshold + 50:
                    print("🎉 表现卓越，提前晋级下一阶段！")
                    break

        # 3.3 阶段结束
        # 保存该阶段的最终模型作为归档
        stage_save_name = os.path.join(log_dir, f"model_stage_{stage_idx}_level{level}")
        model.save(stage_save_name)
        buffer_path = os.path.join(log_dir, f"model_stage_{stage_idx}_level{level}_replay_buffer")
        model.save_replay_buffer(buffer_path)
        env.save(stage_save_name + "_vecnormalize.pkl")

        # 晋级判断
        # 这里可以加逻辑：如果跑完了 target_steps 还是没达到 threshold，
        # 是报错停止，还是降低难度回炉重造？
        # 目前逻辑是直接进入下一关，但你可以加个 while True 锁死。

    print("🏁 所有课程训练完成！")
    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()