import matlab.engine
import time
import argparse
import numpy as np
import os


def main(args):

    state_dim = 3
    action_dim = 1
    action_bound = 2.0
    action_scale = 1.0

    sample_time = 0.05
    stop_time = 20
    step_max = int(stop_time / sample_time)

    batch_size = 256
    auto_entropy = True
    max_episodes = 600

    eng = matlab.engine.start_matlab()
    env_name = 'pendulum'
    eng.load_system(env_name)

    for ep in range(max_episodes):
        t1 = time.time()

        eng.set_param(env_name, 'StopTime', str(21), nargout=0)
        eng.set_param(env_name + '/pause_time', 'value', str(0.01), nargout=0)
        eng.set_param(env_name + '/input', 'value', str(0), nargout=0)
        eng.set_param(env_name, 'SimulationCommand', 'start', nargout=0)
        pause_time = 0.0

        obs_list, action_list, reward_list, done_list = [], [], [], []
        clock_list = []

        for step in range(step_max):
            model_status = eng.get_param(env_name, 'SimulationStatus')
            while(model_status == 'paused'):
                clock = np.array(eng.eval(out.time.Data))[-1]
                obs = np.array(eng.eval(out.obs.Data))[-1]
                reward = np.array(eng.eval(out.reward.Data))[-1]
                action = [0.0]

                clock_list.append(clock)
                obs_list.append(obs)
                action_list.append(action)
                reward_list.append(reward)
                done_list.append(0.0) # pendulum has no done, not added for simplicity

                # 把pause_time每次加上采样时间，实现最后的时序控制
                pause_time += sample_time

                # training process
                if (pause_time + 0.5) > stop_time:
                    done_list[-1] = 1.0
                    eng.set_param(env_name, 'SimulationCommand', 'stop', nargout=0)

                    len_list = len(obs_list)
                    for i1 in range(len_list - 1):
                        obs = obs_list[i1]
                        action = action_list[i1]
                        reward = reward_list[i1 + 1]
                        next_obs = obs_list[i1 + 1]
                        done = done_list[i1 + 1]



