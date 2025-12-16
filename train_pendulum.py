import os
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.monitor import Monitor
from simulink_env import SimulinkGymEnv  # 确保你之前的环境文件叫这个名字


# --- 1. 定义一个新的 Callback 类 ---
class CheckpointWithBufferCallback(BaseCallback):
    """
    兼容 SB3 v1.6.0 的自定义 Callback
    用于同时保存模型和 Replay Buffer
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "sac_step", verbose: int = 0):
        super(CheckpointWithBufferCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # 创建保存目录
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # 1. 保存模型 (.zip)
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(model_path)

            # 2. 保存 Replay Buffer (.pkl) - 这是 v1.6.0 缺少的逻辑
            buffer_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps_replay_buffer")
            # 注意：v1.6.0 中 model.save_replay_buffer 是存在的，只是 Callback 里没封装
            self.model.save_replay_buffer(buffer_path)

            if self.verbose > 1:
                print(f"Saved model and replay buffer to {self.save_path}")
        return True


def main():
    # --- 1. 配置路径与参数 ---
    model_name = 'rlSimplePendulumModel'
    log_dir = "./logs/"
    best_model_dir = os.path.join(log_dir, "best_model")
    checkpoint_dir = os.path.join(log_dir, "checkpoints")

    # 定义最新的检查点路径 (用于断点续训)
    # 我们约定：每次训练结束或中断前，都保存一个名为 "sac_simulink_latest" 的模型
    last_model_path = os.path.join(log_dir, "sac_simulink_latest.zip")
    last_buffer_path = os.path.join(log_dir, "sac_simulink_latest_buffer.pkl")

    # 创建文件夹
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- 2. 创建环境 ---
    # 训练环境
    env = SimulinkGymEnv(model_name, dt=0.05, stop_time=20.0)
    env = Monitor(env, log_dir)  # Monitor 用于记录数据供 TensorBoard 和 EvalCallback 使用

    # 评估环境 (EvalCallback 需要一个独立的环境来测试，防止干扰训练)
    eval_env = SimulinkGymEnv(model_name, dt=0.05, stop_time=20.0,new_process=True)
    eval_env = Monitor(eval_env, log_dir)

    # --- 3. 定义回调函数 (Callbacks) ---

    # A. 早停机制: 当评估奖励 > -740 时停止
    stop_train_callback = StopTrainingOnRewardThreshold(reward_threshold=-740, verbose=1)

    # B. 定期评估 & 保存最优模型
    # eval_freq=2000: 每训练 2000 步 (约 5 个 episode)，暂停训练，用 eval_env 跑几次测试
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=best_model_dir,
                                 log_path=best_model_dir,
                                 eval_freq=2000,
                                 callback_on_new_best=stop_train_callback,  # 达标就触发早停
                                 deterministic=True,  # 使用确定性策略评估
                                 render=False)

    # C. 定期保存 Checkpoint (每 1000 步保存一个，防止断电白跑)
    checkpoint_callback = CheckpointWithBufferCallback(save_freq=1000,
                                             save_path=checkpoint_dir,
                                             name_prefix="sac_step",
                                             verbose=1)

    # 将所有回调组合起来
    callbacks = [eval_callback, checkpoint_callback]

    # --- 4. 断点续训逻辑 (核心) ---
    total_timesteps = 10000  # 总目标步数

    if os.path.exists(last_model_path):
        print(f"检测到上次训练的模型: {last_model_path}，正在加载并继续训练...")

        # 加载模型
        model = SAC.load(last_model_path, env=env, tensorboard_log="./sac_simulink_tb/")

        # 加载 Replay Buffer (这对 SAC 这种 Off-policy 算法非常重要，否则又要重新收集数据)
        if os.path.exists(last_buffer_path):
            print("正在加载 Replay Buffer...")
            model.load_replay_buffer(last_buffer_path)
        else:
            print("警告：未找到对应的buffer文件！训练将从空Buffer开始")

        # reset_num_timesteps=False 表示不要重置步数计数器，接着上次的 Step 继续数
        print("准备继续训练...")
        model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=False)

    else:
        print("未检测到旧模型，开始新的训练...")
        model = SAC("MlpPolicy", env, verbose=1,
                    learning_rate=3e-4,
                    batch_size=256,
                    tensorboard_log="./sac_simulink_tb/")

        model.learn(total_timesteps=total_timesteps, callback=callbacks)

    # --- 5. 训练结束后的保存 ---
    print("训练结束或触发早停。正在保存最终状态...")
    final_save_path = os.path.join(log_dir, "sac_simulink_final")
    model.save(final_save_path)
    model.save_replay_buffer(f"{final_save_path}_replay_buffer")  # 保存经验池以便下次继续

    # 关闭环境
    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()