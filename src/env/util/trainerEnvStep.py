from src.env.util.step import Step
import numpy as np
from src.env.util.baselineTrainerEnvStep import BaselineTrainerEnvStep
import math
from src.env.util.trainerEnvReward import TrainerEnvReward


class TrainerEnvStep(Step):

    def __init__(self, config, baseline_env_step_type, registred_type, reward_type):
        super().__init__(config)
        self.baseline_step = BaselineTrainerEnvStep(config=None, registred_type=baseline_env_step_type)
        self.reward = TrainerEnvReward(config=None, registred_type=reward_type)
        if registred_type is None:
            self.step = self.trainer_env_v4_with_action_dim3_change_step
        elif registred_type == 'ACTION_DIM3_CHANGE_V3':
            self.step = self.trainer_env_v3_with_action_dim3_change_step
        elif registred_type == 'ACTION_DIM3_CHANGE_V4':
            self.step = self.trainer_env_v4_with_action_dim3_change_step
        elif registred_type == 'V4':
            self.step = self.trainer_env_v4_step
        elif registred_type == 'V3':
            self.step = self.trainer_env_v3_step
        else:
            raise IndexError('Not support %s step function type' % registred_type)

    def trainer_env_v4_with_action_dim3_change_step(self, env, action):
        print("Trainer action=", action)
        _, reward, _, info = self.baseline_step.step(env=env, action=action)
        # TODO USE THE VALUE TO GENERATE YOUR OWN OBS, ACTION ..
        ln_cyber = len(env.target_agent_cyber_env_reward_deque)
        ln_real = len(env.target_agent_real_env_reward_deque)
        real_r_his = []
        cyber_r_his = []
        dyna_err_his = []

        for idx in range(ln_cyber - env.td, ln_cyber):
            if idx < 0:
                cyber_r_his.append(env.target_agent_cyber_env_reward_deque[0])
                dyna_err_his.append(env.dyna_error_dequeu[0])
            else:
                cyber_r_his.append(env.target_agent_cyber_env_reward_deque[idx])
                dyna_err_his.append(env.dyna_error_dequeu[idx])

        for idx in range(ln_real - env.td * 2, ln_real, 2):
            if idx < 0:
                real_r_his.append(env.target_agent_real_env_reward_deque[0] * 2)
            else:
                real_r_his.append(
                    env.target_agent_real_env_reward_deque[idx] + env.target_agent_real_env_reward_deque[idx + 1])

        env.real_r_his = env.real_r_his + real_r_his
        env.cyber_r_his = cyber_r_his
        env.dyna_err_his = dyna_err_his
        obs = env._get_obs()

        done = False
        if info[2] is False:
            action = np.array(action).squeeze().tolist()
            action[2] = 1.0
        print("Trainer reward=", reward)
        env.log_file_content.append({
            'INDEX': env.log_print_count,
            'OBS': np.array(obs).squeeze().tolist(),
            'REWARD': float(reward),
            'DONE': done,
            'ACTION': np.array(action).squeeze().tolist(),
            'VALUE_FUNCTION_LOSS': env.critic_loss,
            'CONTROLLER_LOSS': env.actor_loss,
            'VALUE_FUNCTION_LOSS_CHANGE': env.critic_change,
            'CONTROLLER_LOSS_CHANGE': env.actor_change,
            'TARGET_AGENT_REWARD_OLD': info[1],
            'TARGET_AGENT_REWARD_NEW': info[0],
            'TARGET_AGENT_TRULY_TRAIN_CYBER': info[2]
        })
        env.log_print_count += 1

        return obs, reward, done, info

    def trainer_env_v3_with_action_dim3_change_step(self, env, action):
        print("Trainer action=", action)
        _, reward, _, info = self.baseline_step.step(env=env, action=action)
        # TODO USE THE VALUE TO GENERATE YOUR OWN OBS, ACTION ..
        ln_cyber = len(env.target_agent_cyber_env_reward_deque)
        ln_real = len(env.target_agent_real_env_reward_deque)
        real_r_his = []
        cyber_r_his = []
        dyna_err_his = []

        for idx in range(ln_cyber - env.td, ln_cyber):
            if idx < 0:
                cyber_r_his.append(env.target_agent_cyber_env_reward_deque[0])
                dyna_err_his.append(env.dyna_error_dequeu[0])
            else:
                cyber_r_his.append(env.target_agent_cyber_env_reward_deque[idx])
                dyna_err_his.append(env.dyna_error_dequeu[idx])

        for idx in range(ln_real - env.td * 2, ln_real, 2):
            if idx < 0:
                real_r_his.append(env.target_agent_real_env_reward_deque[0] * 2)
            else:
                real_r_his.append(
                    env.target_agent_real_env_reward_deque[idx] + env.target_agent_real_env_reward_deque[idx + 1])

        env.real_r_his = env.real_r_his + real_r_his
        env.cyber_r_his = cyber_r_his
        env.dyna_err_his = dyna_err_his
        obs = env._get_obs()
        reward = self.reward(env=env)

        done = False
        if info[2] is False:
            action = np.array(action).squeeze().tolist()
            action[2] = 1.0
        print("Trainer reward=", reward)
        env.log_file_content.append({
            'INDEX': env.log_print_count,
            'OBS': np.array(obs).squeeze().tolist(),
            'REWARD': float(reward),
            'DONE': done,
            'ACTION': np.array(action).squeeze().tolist(),
            'VALUE_FUNCTION_LOSS': env.critic_loss,
            'CONTROLLER_LOSS': env.actor_loss,
            'VALUE_FUNCTION_LOSS_CHANGE': env.critic_change,
            'CONTROLLER_LOSS_CHANGE': env.actor_change,
            'TARGET_AGENT_REWARD_OLD': info[1],
            'TARGET_AGENT_REWARD_NEW': info[0],
            'TARGET_AGENT_TRULY_TRAIN_CYBER': info[2]
        })
        env.log_print_count += 1

        return obs, reward, done, info

    def trainer_env_v4_step(self, env, action):
        print("Trainer action=", action)
        _, reward, _, info = self.baseline_step.step(env=env, action=action)
        # TODO USE THE VALUE TO GENERATE YOUR OWN OBS, ACTION ..
        ln_cyber = len(env.target_agent_cyber_env_reward_deque)
        ln_real = len(env.target_agent_real_env_reward_deque)
        real_r_his = []
        cyber_r_his = []
        dyna_err_his = []

        for idx in range(ln_cyber - env.td, ln_cyber):
            if idx < 0:
                cyber_r_his.append(env.target_agent_cyber_env_reward_deque[0])
                dyna_err_his.append(env.dyna_error_dequeu[0])
            else:
                cyber_r_his.append(env.target_agent_cyber_env_reward_deque[idx])
                dyna_err_his.append(env.dyna_error_dequeu[idx])

        for idx in range(ln_real - env.td * 2, ln_real, 2):
            if idx < 0:
                real_r_his.append(env.target_agent_real_env_reward_deque[0] * 2)
            else:
                real_r_his.append(
                    env.target_agent_real_env_reward_deque[idx] + env.target_agent_real_env_reward_deque[idx + 1])

        env.real_r_his = env.real_r_his + real_r_his
        env.cyber_r_his = cyber_r_his
        env.dyna_err_his = dyna_err_his
        obs = env._get_obs()

        done = False
        info[2] = True
        print("Trainer reward=", reward)
        env.log_file_content.append({
            'INDEX': env.log_print_count,
            'OBS': np.array(obs).squeeze().tolist(),
            'REWARD': float(reward),
            'DONE': done,
            'ACTION': np.array(action).squeeze().tolist(),
            'VALUE_FUNCTION_LOSS': env.critic_loss,
            'CONTROLLER_LOSS': env.actor_loss,
            'VALUE_FUNCTION_LOSS_CHANGE': env.critic_change,
            'CONTROLLER_LOSS_CHANGE': env.actor_change,
            'TARGET_AGENT_REWARD_OLD': info[1],
            'TARGET_AGENT_REWARD_NEW': info[0],
            'TARGET_AGENT_TRULY_TRAIN_CYBER': info[2]
        })
        env.log_print_count += 1

        return obs, reward, done, info

    def trainer_env_v3_step(self, env, action):
        print("Trainer action=", action)
        _, reward, _, info = self.baseline_step.step(env=env, action=action)
        # TODO USE THE VALUE TO GENERATE YOUR OWN OBS, ACTION ..
        ln_cyber = len(env.target_agent_cyber_env_reward_deque)
        ln_real = len(env.target_agent_real_env_reward_deque)
        real_r_his = []
        cyber_r_his = []
        dyna_err_his = []

        for idx in range(ln_cyber - env.td, ln_cyber):
            if idx < 0:
                cyber_r_his.append(env.target_agent_cyber_env_reward_deque[0])
                dyna_err_his.append(env.dyna_error_dequeu[0])
                real_r_his.append(env.target_agent_real_env_reward_deque[0])
            else:
                cyber_r_his.append(env.target_agent_cyber_env_reward_deque[idx])
                dyna_err_his.append(env.dyna_error_dequeu[idx])
                real_r_his.append(env.target_agent_real_env_reward_deque[idx])

        env.real_r_his = env.real_r_his + real_r_his
        env.cyber_r_his = cyber_r_his
        env.dyna_err_his = dyna_err_his
        obs = env._get_obs()
        reward = self.reward(env=env)
        done = False
        info[2] = True
        print("Trainer reward=", reward)
        env.log_file_content.append({
            'INDEX': env.log_print_count,
            'OBS': np.array(obs).squeeze().tolist(),
            'REWARD': float(reward),
            'DONE': done,
            'ACTION': np.array(action).squeeze().tolist(),
            'VALUE_FUNCTION_LOSS': env.critic_loss,
            'CONTROLLER_LOSS': env.actor_loss,
            'VALUE_FUNCTION_LOSS_CHANGE': env.critic_change,
            'CONTROLLER_LOSS_CHANGE': env.actor_change,
            'TARGET_AGENT_REWARD_OLD': info[1],
            'TARGET_AGENT_REWARD_NEW': info[0],
            'TARGET_AGENT_TRULY_TRAIN_CYBER': info[2]
        })
        env.log_print_count += 1

        return obs, reward, done, info

    def trainer_env_v3_step_with_reward_differ_real_cyber_mean_reward(self, env, action):
        obs, reward, done, info = self.trainer_env_v3_step(env=env, action=action)
        obs = (env.real_r_his[-1], env.cyber_r_his[-1])
        return obs, reward, done, info
