import os
import glob
import numpy as np
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.monitor import Monitor

# 导入你的环境类
from simulink_env import SimulinkGymEnv


# ==================== 1. 辅助类与函数 ====================

class CheckpointWithBufferCallback(BaseCallback):
    """
    兼容 SB3 v1.6.0 的自定义 Callback，用于保存模型和 Buffer
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "sac_step", verbose: int = 0):
        super(CheckpointWithBufferCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # 保存模型
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(model_path)
            # 保存 Buffer
            buffer_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps_replay_buffer")
            self.model.save_replay_buffer(buffer_path)
            if self.verbose > 1:
                print(f"Saved model and buffer to {self.save_path}")
        return True


def find_latest_checkpoint(log_dir, prefix="sac_step"):
    """自动查找步数最大的模型文件"""
    # 查找 log_dir/checkpoints/ 下的所有 zip 文件
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        return None, 0

    search_pattern = os.path.join(ckpt_dir, f"{prefix}_*_steps.zip")
    files = glob.glob(search_pattern)
    if not files:
        return None, 0

    try:
        # 解析文件名: sac_step_10000_steps.zip -> 10000
        latest_file = max(files, key=lambda x: int(x.split('_steps')[0].split('_')[-1]))
        latest_step = int(latest_file.split('_steps')[0].split('_')[-1])
        return latest_file, latest_step
    except:
        return None, 0


def collect_expert_data(env, model, n_episodes=10):
    """
    并在多环境中收集专家数据 (全油门直行)
    """
    print(f"正在收集专家数据 ({n_episodes} 批次)...")

    # 获取环境数量 (并行环境下 env.num_envs > 1)
    n_envs = env.num_envs

    # 专家动作: [油门=1.0, 转向=0.0]
    # 扩展为 (n_envs, 2) 的矩阵，因为并行环境需要同时接收所有环境的动作
    expert_action = np.tile([1.0, 0.0], (n_envs, 1))

    # 简单的计数逻辑
    episodes_collected = 0
    obs = env.reset()

    while episodes_collected < n_episodes:
        # 执行动作
        next_obs, rewards, dones, infos = env.step(expert_action)

        # 将数据存入 Buffer
        # SB3 的 add 方法会自动处理 VecEnv 的数据维度
        model.replay_buffer.add(obs, next_obs, expert_action, rewards, dones, infos)

        obs = next_obs

        # 统计完成的 episode 数量 (如果任意一个环境 done 了)
        if np.any(dones):
            episodes_collected += np.sum(dones)

    print(f"专家数据收集完成！Buffer 大小: {model.replay_buffer.pos}")


# ==================== 2. 主函数 ====================

def main():
    # --- 配置 ---
    model_name = 'RLmodel'
    log_dir = "./logs/"
    best_model_dir = os.path.join(log_dir, "best_model")
    checkpoint_dir = os.path.join(log_dir, "checkpoints")

    # 并行核心数设置 (建议设置为 CPU 物理核心数 - 2)
    # 注意：每个进程都会启动一个 MATLAB，请确保内存充足 (16GB内存建议设为 4)
    N_ENVS = 4

    # 创建文件夹
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- 创建并行训练环境 ---
    # 定义环境参数
    # debug_mode=False 确保启动新的独立 MATLAB 进程
    env_kwargs = {'model_name': model_name, 'dt': 0.05, 'stop_time': 20.0, 'debug_mode': False}

    print(f"正在启动 {N_ENVS} 个并行 MATLAB 环境 (可能需要几分钟)...")
    env = make_vec_env(
        lambda: SimulinkGymEnv(**env_kwargs),
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv  # <--- 使用多进程类
    )
    env = VecMonitor(env, log_dir)  # 并行环境下使用 VecMonitor

    # --- 创建评估环境 (单进程) ---
    print("正在启动评估环境 (单进程带界面)...")
    # 评估环境只需一个，debug_mode=False 强制启动独立进程避免干扰
    eval_env = SimulinkGymEnv(model_name, dt=0.05, stop_time=20.0, debug_mode=True)
    eval_env = Monitor(eval_env, log_dir)

    # --- Callbacks ---
    stop_train_callback = StopTrainingOnRewardThreshold(reward_threshold=700, verbose=1)

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=best_model_dir,
                                 log_path=best_model_dir,
                                 # eval_freq 需要除以并行数，因为 step 是并行的
                                 eval_freq=2000 // N_ENVS,
                                 callback_on_new_best=stop_train_callback,
                                 deterministic=True,
                                 render=False)

    checkpoint_callback = CheckpointWithBufferCallback(
        # 同样调整保存频率
        save_freq=1000 // N_ENVS,
        save_path=checkpoint_dir,
        name_prefix="sac_step",
        verbose=1)

    callbacks = [eval_callback, checkpoint_callback]

    # --- 断点续训逻辑 ---
    total_timesteps = 100000

    # 自动查找最新的 checkpoint
    # latest_model_path, latest_step_count = find_latest_checkpoint(log_dir, "sac_step")

    if False:
        print(f"\n=== 检测到中断的训练 ===")
        print(f"加载模型: {latest_model_path}")
        print(f"已完成步数: {latest_step_count}")

        # 1. 加载模型
        model = SAC.load(latest_model_path, env=env, tensorboard_log="./sac_simulink_tb/")

        # 2. 推断 Buffer 路径并加载
        # 假设文件名格式: sac_step_10000_steps.zip -> sac_step_10000_steps_replay_buffer.pkl
        buffer_path = latest_model_path.replace(".zip", "_replay_buffer.pkl")

        if os.path.exists(buffer_path):
            print(f"正在加载经验回放池: {buffer_path}")
            model.load_replay_buffer(buffer_path)
        else:
            print("警告：未找到 Buffer 文件，将从空 Buffer 继续训练！")

        model.num_timesteps = latest_step_count
        remaining_steps = total_timesteps - latest_step_count

        if remaining_steps > 0:
            print(f"继续训练剩余 {remaining_steps} 步...")
            model.learn(total_timesteps=remaining_steps, callback=callbacks, reset_num_timesteps=False)
        else:
            print("训练已达到目标步数。")

    else:
        print("\n=== 开始新的训练 ===")
        model = SAC("MlpPolicy", env, verbose=1,
                    learning_rate=3e-4,
                    batch_size=256,
                    tensorboard_log="./sac_simulink_tb/")

        # 收集专家数据 (利用并行环境加速收集)
        # 跑 20 个 episodes，因为有 4 个环境，实际上每轮跑 4 个，只需跑 5 轮
        # collect_expert_data(eval_env, model, n_episodes=5)
        collect_expert_data(env, model, n_episodes=20)

        model.learn(total_timesteps=total_timesteps, callback=callbacks)

    # --- 结束保存 ---
    print("训练结束。保存最终模型...")
    final_path = os.path.join(log_dir, "sac_simulink_final")
    model.save(final_path)
    model.save_replay_buffer(f"{final_path}_replay_buffer")

    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()