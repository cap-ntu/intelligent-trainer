#!/usr/bin/env python3

'''
Main script to run the tests for single-head trainer, example:

Test random trainer:
    run_multiple_experiments(game_env_name='Swimmer-v1',
                             cuda_device=0,
                             num=1,
                             config_set_path=CONFIG_SET_SWIMMER_INTEL_REINFORCE,
                             model_config_dict=MODEL_NET_WORK_CONFIG_DICT_SWIMMER_INTEL_REINFORCE,
                             target_model_type='TRPO',
                             intelligent_trainer_model_type='RANDOM',   #'DQN', 'REINFORCE' , 'FIX', 'RANDOM'
                             exp_end_with='_debug')

Test DQN trainer:
    run_multiple_experiments(game_env_name='Swimmer-v1',
                             cuda_device=0,
                             num=1,
                             config_set_path=CONFIG_SET_SWIMMER_INTEL,
                             model_config_dict=MODEL_NET_WORK_CONFIG_DICT_SWIMMER_INTEL,
                             target_model_type='TRPO',
                             intelligent_trainer_model_type='DQN',  # 'DQN', 'REINFORCE' , 'FIX', 'RANDOM'
                             exp_end_with='_debug')

Test REINFORCE trainer:
    run_multiple_experiments(game_env_name='Swimmer-v1',
                             cuda_device=0,
                             num=1,
                             config_set_path=CONFIG_SET_SWIMMER_INTEL,
                             model_config_dict=MODEL_NET_WORK_CONFIG_DICT_SWIMMER_INTEL,
                             target_model_type='TRPO',
                             intelligent_trainer_model_type='REINFORCE',  # 'DQN', 'REINFORCE' , 'FIX', 'RANDOM'
                             exp_end_with='_debug')

Test FIX Trainer (model-based RL with fixed action 0.6):
    run_multiple_experiments(game_env_name='Swimmer-v1',
                             cuda_device=0,
                             num=1,
                             config_set_path=CONFIG_SET_SWIMMER_INTEL,
                             model_config_dict=MODEL_NET_WORK_CONFIG_DICT_SWIMMER_INTEL,
                             target_model_type='TRPO',
                             intelligent_trainer_model_type='FIX',  # 'DQN', 'REINFORCE' , 'FIX', 'RANDOM'
                             exp_end_with='_debug')



Refer __main__ function for other cases
'''



import os
import sys

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)
PAR_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
sys.path.append(PAR_PATH)
from src.env.util import *
import util.utilNew as ut_new
from conf.envBound import get_bound_file
import tensorflow as tf
from src.util.plotter import Plotter
from util.utilNew import create_tmp_config_file
import config as cfg
import time


def run_single_experiment(game_env_name, cuda_device, config_set_path, model_config_dict, target_model_type,
                          intelligent_trainer_model_type, seed=None, exp_end_with=''):
    tf.set_random_seed(seed)
    player, sess = ut_new.create_intelligent_game(cost_fn=COST_FUNCTION_ENV_DICT[game_env_name],
                                                  env_id=game_env_name,
                                                  game_specific_config_path=model_config_dict[game_env_name],
                                                  config_set_path=config_set_path,
                                                  cuda_device=cuda_device,
                                                  bound_file=get_bound_file(env_name=game_env_name),
                                                  reset_fn=RESET_FUNCTION_ENV_DICT[game_env_name],
                                                  done_fn=DONE_FUNCTION_ENV_DICT[game_env_name],
                                                  target_model_type=target_model_type,
                                                  intelligent_model_type=intelligent_trainer_model_type,
                                                  exp_end_with=exp_end_with)
    try:
        player.play(seed)
        player.print_log_to_file()
        player.save_all_model()
    except KeyboardInterrupt:
        player.print_log_to_file()
        player.save_all_model()


def run_multiple_experiments(game_env_name, cuda_device, num, config_set_path, model_config_dict, target_model_type,
                             intelligent_trainer_model_type,
                             seed=None, exp_end_with=''):
    log_dir_path = []
    tmp_path = create_tmp_config_file(game_env_name=game_env_name, orgin_config_path=config_set_path)
    cfg.config_dict = {
        'NOT_TRPO_CLEAR_MEMORY': False,
        'STE_V3_TEST_MOVE_OUT': False,
        'F1=0': False,
        'F2=0': False,
        # 'SWIMMER_HORIZON': 50
        "TIME_SEED": True,
        'SPLIT_COUNT': 2,
        "TRAINER_ENV_STATE_AGENT_STEP_COUNT": True
    }

    seed = [i for i in range(num)]
    for i in range(num):
        if "TIME_SEED" in cfg.config_dict and cfg.config_dict['TIME_SEED'] is True:
            seed[i] = int(round(time.time() * 1000)) % (2 ** 32 - 1)
        tf.reset_default_graph()
        tf.set_random_seed(seed[i])
        player, sess = ut_new.create_intelligent_game(cost_fn=COST_FUNCTION_ENV_DICT[game_env_name],
                                                      env_id=game_env_name,
                                                      game_specific_config_path=tmp_path,
                                                      config_set_path=tmp_path,
                                                      cuda_device=cuda_device,
                                                      bound_file=get_bound_file(env_name=game_env_name),
                                                      done_fn=DONE_FUNCTION_ENV_DICT[game_env_name],
                                                      reset_fn=RESET_FUNCTION_ENV_DICT[game_env_name],
                                                      target_model_type=target_model_type,
                                                      intelligent_model_type=intelligent_trainer_model_type,
                                                      exp_end_with=exp_end_with)

        log_dir_path.append(player.logger.log_dir)

        try:
            player.play(seed[i])
            player.print_log_to_file()
            player.save_all_model()
        except KeyboardInterrupt:
            player.print_log_to_file()
            player.save_all_model()
            # TODO fix bug for load model
            # player.load_all_model()
        sess = tf.get_default_session()
        if sess:
            sess.__exit__(None, None, None)
    for log in log_dir_path:
        print(log)
    Plotter.plot_multiply_target_agent_reward(path_list=log_dir_path)


if __name__ == '__main__':
    from conf.configSet_MountainCarContinuous_Intel import CONFIG_SET_MOUNTAIN_CAR_CONTINUOUS_CONFIG_INTEL, \
        MODEL_NET_WORK_CONFIG_DICT_MOUNTAIN_CAR_CONTINUOUS_INTEL
    from conf.configSet_Reacher_Intel import CONFIG_SET_REACHER_INTEL, MODEL_NET_WORK_CONFIG_DICT_REACHER_INTEL
    from conf.configSet_HalfCheetah_Intel import CONFIG_SET_HALFCHEETAH_INTEL, \
        MODEL_NET_WORK_CONFIG_DICT_HALFCHEETAH_INTEL
    from conf.configSet_Swimmer_Intel import CONFIG_SET_SWIMMER_INTEL, MODEL_NET_WORK_CONFIG_DICT_SWIMMER_INTEL
    from conf.configSet_Pendulum_Intel import CONFIG_SET_PENDULUM_INTEL, MODEL_NET_WORK_CONFIG_DICT_PENDULUM_INTEL


    from conf.configSet_MountainCarContinuous_Intel_REINFORCE import CONFIG_SET_MOUNTAIN_CAR_CONTINUOUS_CONFIG_INTEL_REINFORCE, \
        MODEL_NET_WORK_CONFIG_DICT_MOUNTAIN_CAR_CONTINUOUS_INTEL_REINFORCE
    from conf.configSet_Reacher_Intel_REINFORCE import CONFIG_SET_REACHER_INTEL_REINFORCE, MODEL_NET_WORK_CONFIG_DICT_REACHER_INTEL_REINFORCE
    from conf.configSet_HalfCheetah_Intel_REINFORCE import CONFIG_SET_HALFCHEETAH_INTEL_REINFORCE, \
        MODEL_NET_WORK_CONFIG_DICT_HALFCHEETAH_INTEL_REINFORCE
    from conf.configSet_Swimmer_Intel_REINFORCE import CONFIG_SET_SWIMMER_INTEL_REINFORCE, MODEL_NET_WORK_CONFIG_DICT_SWIMMER_INTEL_REINFORCE
    from conf.configSet_Pendulum_Intel_REINFORCE import CONFIG_SET_PENDULUM_INTEL_REINFORCE, MODEL_NET_WORK_CONFIG_DICT_PENDULUM_INTEL_REINFORCE


    # Pendulum-v0
    # run_multiple_experiments(game_env_name='Pendulum-v0',
    #                          cuda_device=0,
    #                          config_set_path=CONFIG_SET_PENDULUM_INTEL,
    #                          model_config_dict=MODEL_NET_WORK_CONFIG_DICT_PENDULUM_INTEL,
    #                          num=2,
    #                          target_model_type='DDPG',
    #                          intelligent_trainer_model_type='DQN',
    #                          # exp_end_with='_random_fixednew_v5')
    #                          exp_end_with='_intel_v5_1_trainer_state_is_sample_count')

    # MountainCarContinuous-v0
    # run_multiple_experiments(game_env_name='MountainCarContinuous-v0',
    #                          cuda_device=2,
    #                          num=2,
    #                          config_set_path=CONFIG_SET_MOUNTAIN_CAR_CONTINUOUS_CONFIG_INTEL,
    #                          model_config_dict=MODEL_NET_WORK_CONFIG_DICT_MOUNTAIN_CAR_CONTINUOUS_INTEL,
    #                          target_model_type='DDPG',
    #                          intelligent_trainer_model_type='DQN',
    #                          # exp_end_with='_intel_v5_trainer_eps_greedy_30000')
    #                          exp_end_with='_intel_v5_1_trainer_state_is_sample_count')

    # # Reacher-v1
    run_multiple_experiments(game_env_name='Reacher-v1',
                             cuda_device=1,
                             num=1,
                             config_set_path=CONFIG_SET_REACHER_INTEL,
                             model_config_dict=MODEL_NET_WORK_CONFIG_DICT_REACHER_INTEL,
                             target_model_type='TRPO',
                             intelligent_trainer_model_type='FIX',
                             exp_end_with='_debug_FIX')

    # HalfCheetah
    # run_multiple_experiments(game_env_name='HalfCheetah',
    #                          cuda_device=0,
    #                          num=1,
    #                          config_set_path=CONFIG_SET_HALFCHEETAH_INTEL,
    #                          model_config_dict=MODEL_NET_WORK_CONFIG_DICT_HALFCHEETAH_INTEL,
    #                          target_model_type='TRPO',
    #                          intelligent_trainer_model_type='DQN',
    #                          exp_end_with='_intel_v5_1_trainer_state_is_sample_count')
    # # # Swimmer
    # run_multiple_experiments(game_env_name='Swimmer-v1',
    #                          cuda_device=0,
    #                          num=1,
    #                          config_set_path=CONFIG_SET_SWIMMER_INTEL_REINFORCE,
    #                          model_config_dict=MODEL_NET_WORK_CONFIG_DICT_SWIMMER_INTEL_REINFORCE,
    #                          target_model_type='TRPO',
    #                          intelligent_trainer_model_type='RANDOM',   #'DQN', 'REINFORCE' , 'FIX', 'RANDOM'
    #                          exp_end_with='_debug')
    #
    # run_multiple_experiments(game_env_name='Swimmer-v1',
    #                          cuda_device=0,
    #                          num=1,
    #                          config_set_path=CONFIG_SET_SWIMMER_INTEL,
    #                          model_config_dict=MODEL_NET_WORK_CONFIG_DICT_SWIMMER_INTEL,
    #                          target_model_type='TRPO',
    #                          intelligent_trainer_model_type='DQN',  # 'DQN', 'REINFORCE' , 'FIX', 'RANDOM'
    #                          exp_end_with='_debug')
    #
    # run_multiple_experiments(game_env_name='Swimmer-v1',
    #                          cuda_device=0,
    #                          num=1,
    #                          config_set_path=CONFIG_SET_SWIMMER_INTEL,
    #                          model_config_dict=MODEL_NET_WORK_CONFIG_DICT_SWIMMER_INTEL,
    #                          target_model_type='TRPO',
    #                          intelligent_trainer_model_type='REINFORCE',  # 'DQN', 'REINFORCE' , 'FIX', 'RANDOM'
    #                          exp_end_with='_debug')
    #
    # run_multiple_experiments(game_env_name='Swimmer-v1',
    #                          cuda_device=0,
    #                          num=1,
    #                          config_set_path=CONFIG_SET_SWIMMER_INTEL,
    #                          model_config_dict=MODEL_NET_WORK_CONFIG_DICT_SWIMMER_INTEL,
    #                          target_model_type='TRPO',
    #                          intelligent_trainer_model_type='FIX',  # 'DQN', 'REINFORCE' , 'FIX', 'RANDOM'
    #                          exp_end_with='_debug')
