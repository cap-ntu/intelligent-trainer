#!/usr/bin/env python3
'''
Main script to run the tests for NoCyber (equal to original DRL), example:
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
from conf.configSet_MountainCarContinuous_Intel_No_Dyna import \
        CONFIG_SET_MOUNTAIN_CAR_CONTINUOUS_CONFIG_INTEL_NO_DYNA, \
        MODEL_NET_WORK_CONFIG_DICT_MOUNTAIN_CAR_CONTINUOUS_INTEL_NO_DYNA
from conf.configSet_Reacher_Intel_No_Dyna import CONFIG_SET_REACHER_INTEL_NO_DYNA, \
    MODEL_NET_WORK_CONFIG_DICT_REACHER_INTEL_NO_DYNA
from conf.configSet_HalfCheetah_Intel_No_Dyna import CONFIG_SET_HALFCHEETAH_INTEL_NO_DYNA, \
    MODEL_NET_WORK_CONFIG_DICT_HALFCHEETAH_INTEL_NO_DYNA
from conf.configSet_Swimmer_Intel_No_Dyna import CONFIG_SET_SWIMMER_INTEL_NO_DYNA, \
    MODEL_NET_WORK_CONFIG_DICT_SWIMMER_INTEL_NO_DYNA
from conf.configSet_Pendulum_Intel_No_Dyna import CONFIG_SET_PENDULUM_INTEL_NO_DYNA, \
    MODEL_NET_WORK_CONFIG_DICT_PENDULUM_INTEL_NO_DYNA


def run_multiple_experiments(game_env_name, cuda_device, num, config_set_path, model_config_dict, target_model_type,
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
        player, sess = ut_new.create_baseline_game(cost_fn=COST_FUNCTION_ENV_DICT[game_env_name],
                                                   env_id=game_env_name,
                                                   game_specific_config_path=tmp_path,
                                                   config_set_path=tmp_path,
                                                   cuda_device=cuda_device,
                                                   bound_file=get_bound_file(env_name=game_env_name),
                                                   done_fn=DONE_FUNCTION_ENV_DICT[game_env_name],
                                                   reset_fn=RESET_FUNCTION_ENV_DICT[game_env_name],
                                                   target_model_type=target_model_type,
                                                   exp_log_end=exp_end_with)

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
    Plotter.plot_multiply_target_agent_reward(path_list=log_dir_path, fig_id=1)


model_type_dict = {
    "Pendulum-v0": 'DDPG',
    "MountainCarContinuous-v0": 'DDPG',
    "Reacher-v1": 'TRPO',
    "HalfCheetah": 'TRPO',
    "Swimmer-v1": 'TRPO',
}

env_config_dict = {
    "Pendulum-v0": (CONFIG_SET_PENDULUM_INTEL_NO_DYNA, MODEL_NET_WORK_CONFIG_DICT_PENDULUM_INTEL_NO_DYNA),
    "MountainCarContinuous-v0": (CONFIG_SET_MOUNTAIN_CAR_CONTINUOUS_CONFIG_INTEL_NO_DYNA, MODEL_NET_WORK_CONFIG_DICT_MOUNTAIN_CAR_CONTINUOUS_INTEL_NO_DYNA),
    "Reacher-v1": (CONFIG_SET_REACHER_INTEL_NO_DYNA, MODEL_NET_WORK_CONFIG_DICT_REACHER_INTEL_NO_DYNA),
    "HalfCheetah": (CONFIG_SET_HALFCHEETAH_INTEL_NO_DYNA, MODEL_NET_WORK_CONFIG_DICT_HALFCHEETAH_INTEL_NO_DYNA),
    "Swimmer-v1": (CONFIG_SET_SWIMMER_INTEL_NO_DYNA, MODEL_NET_WORK_CONFIG_DICT_SWIMMER_INTEL_NO_DYNA),
}

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str)
parser.add_argument('--cuda_id', type=int, default=0)
parser.add_argument('--num', type=int, default=1)

if __name__ == '__main__':

    args = parser.parse_args()
    print(args)
    run_multiple_experiments(game_env_name=args.env,
                             cuda_device=args.cuda_id,
                             config_set_path=env_config_dict[args.env][0],
                             model_config_dict=env_config_dict[args.env][1],
                             num=args.num,
                             target_model_type=model_type_dict[args.env])
