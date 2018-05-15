#!/usr/bin/env python3

import os
import sys

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)
PAR_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
sys.path.append(PAR_PATH)
from src.env.costDoneFunction import COST_FUNCTION_ENV_DICT, DONE_FUNCTION_ENV_DICT
from src.env.utils import GAME_ENV_NAME_DICT
import util.utilNew as ut_new
from config.envBound import get_bound_file
import tensorflow as tf
from src.util.plotter import Plotter
import argparse


def run_single_experiment(game_env_name, cuda_device, config_set_path, model_config_dict, target_model_type,
                          intelligent_trainer_model_type, seed=None):
    player, sess = ut_new.create_intelligent_game(cost_fn=COST_FUNCTION_ENV_DICT[game_env_name],
                                                  env_id=game_env_name,
                                                  game_specific_config_path=model_config_dict[game_env_name],
                                                  config_set_path=config_set_path,
                                                  cuda_device=cuda_device,
                                                  bound_file=get_bound_file(env_name=game_env_name),
                                                  done_fn=DONE_FUNCTION_ENV_DICT[game_env_name],
                                                  target_model_type=target_model_type,
                                                  intelligent_model_type=intelligent_trainer_model_type)
    try:
        player.play(seed)
        player.print_log_to_file()
        player.save_all_model()
    except KeyboardInterrupt:
        player.print_log_to_file()
        player.save_all_model()


def run_multiple_experiments(game_env_name, cuda_device, num, config_set_path, model_config_dict, target_model_type,
                             intelligent_trainer_model_type,
                             seed=None):
    log_dir_path = []
    if seed is None:
        seed = [0, 1, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222, 333, 444]
    for i in range(num):
        tf.reset_default_graph()
        player, sess = ut_new.create_intelligent_game(cost_fn=COST_FUNCTION_ENV_DICT[game_env_name],
                                                      env_id=game_env_name,
                                                      game_specific_config_path=model_config_dict[game_env_name],
                                                      config_set_path=config_set_path,
                                                      cuda_device=cuda_device,
                                                      bound_file=get_bound_file(env_name=game_env_name),
                                                      done_fn=DONE_FUNCTION_ENV_DICT[game_env_name],
                                                      target_model_type=target_model_type,
                                                      intelligent_model_type=intelligent_trainer_model_type)

        log_dir_path.append(player.logger.log_dir)

        try:
            player.play(seed[i])
            player.print_log_to_file()
            player.save_all_model()
        except KeyboardInterrupt:
            player.print_log_to_file()
            player.save_all_model()
        sess = tf.get_default_session()
        if sess:
            sess.__exit__(None, None, None)
    for log in log_dir_path:
        print(log)
    # Plotter.plot_multiply_target_agent_reward(path_list=log_dir_path)


if __name__ == '__main__':
    from config.configSet_MountainCarContinuous_Intel import CONFIG_SET_MOUNTAIN_CAR_CONTINUOUS_CONFIG_INTEL, \
        MODEL_NET_WORK_CONFIG_DICT_MOUNTAIN_CAR_CONTINUOUS_INTEL
    from config.configSet_Pendulum_Intel import CONFIG_SET_PENDULUM_INTEL, MODEL_NET_WORK_CONFIG_DICT_PENDULUM_INTEL

    from config.configSet_HalfCheetah_Intel import CONFIG_SET_HALFCHEETAH_INTEL, \
        MODEL_NET_WORK_CONFIG_DICT_HALFCHEETAH_INTEL
    from config.configSet_Swimmer_Intel import CONFIG_SET_SWIMMER_INTEL, MODEL_NET_WORK_CONFIG_DICT_SWIMMER_INTEL
    from config.configSet_Reacher_Intel import CONFIG_SET_REACHER_INTEL, MODEL_NET_WORK_CONFIG_DICT_REACHER_INTEL

    parser = argparse.ArgumentParser()
    parser.add_argument('env',
                        help='Environment name, could be: Pendulum-v0, MountainCarContinuous-v0, Reacher-v1, HalfCheetah, Swimmer-v1')
    env_name = parser.parse_args().env
    if env_name == 'Pendulum-v0':
        run_multiple_experiments(game_env_name='Pendulum-v0',
                                 cuda_device=0,
                                 config_set_path=CONFIG_SET_PENDULUM_INTEL,
                                 model_config_dict=MODEL_NET_WORK_CONFIG_DICT_PENDULUM_INTEL,
                                 num=10,
                                 target_model_type='DDPG',
                                 intelligent_trainer_model_type='DQN')
    elif env_name == 'MountainCarContinuous-v0':
        run_multiple_experiments(game_env_name='MountainCarContinuous-v0',
                                 cuda_device=0,
                                 num=10,
                                 config_set_path=CONFIG_SET_MOUNTAIN_CAR_CONTINUOUS_CONFIG_INTEL,
                                 model_config_dict=MODEL_NET_WORK_CONFIG_DICT_MOUNTAIN_CAR_CONTINUOUS_INTEL,
                                 target_model_type='DDPG',
                                 intelligent_trainer_model_type='DQN')
    elif env_name == 'Reacher-v1':
        run_multiple_experiments(game_env_name='Reacher-v1',
                                 cuda_device=0,
                                 num=10,
                                 config_set_path=CONFIG_SET_REACHER_INTEL,
                                 model_config_dict=MODEL_NET_WORK_CONFIG_DICT_REACHER_INTEL,
                                 target_model_type='TRPO',
                                 intelligent_trainer_model_type='DQN')
    elif env_name == 'Swimmer-v1':
        run_multiple_experiments(game_env_name='Swimmer-v1',
                                 cuda_device=0,
                                 num=10,
                                 config_set_path=CONFIG_SET_SWIMMER_INTEL,
                                 model_config_dict=MODEL_NET_WORK_CONFIG_DICT_SWIMMER_INTEL,
                                 target_model_type='TRPO',
                                 intelligent_trainer_model_type='DQN')
    elif env_name == 'HalfCheetah':

        run_multiple_experiments(game_env_name='HalfCheetah',
                                 cuda_device=0,
                                 num=10,
                                 config_set_path=CONFIG_SET_HALFCHEETAH_INTEL,
                                 model_config_dict=MODEL_NET_WORK_CONFIG_DICT_HALFCHEETAH_INTEL,
                                 target_model_type='TRPO',
                                 intelligent_trainer_model_type='DQN')

    else:
        raise KeyError(
            'Wrong env name, please use one of: Pendulum-v0, MountainCarContinuous-v0, Reacher-v1, HalfCheetah, Swimmer-v1')
