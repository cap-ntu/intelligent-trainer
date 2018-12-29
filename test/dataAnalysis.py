import numpy as np
import math
import json
from src.util.plotter import Plotter
from src.config.config import Config
import os


def compute_best_eps_reward(test_file, eps_size=100):
    average_reward = 0.0
    reward_list = []
    real_sample_used_list = []
    cyber_sample_used_list = []
    count = 0
    with open(file=test_file, mode='r') as f:
        train_data = json.load(fp=f)
        for sample in train_data:
            # if count % 2 == 0:
            #     print(sample['REWARD_SUM'])
            #     pass
            # else:
            #     reward_list.append(sample['REWARD_SUM'])
            #     real_sample_used_list.append(sample['REAL_SAMPLE_COUNT'])
            #     cyber_sample_used_list.append(sample['CYBER_SAMPLE_COUNT'])

            reward_list.append(sample['REWARD_SUM'])
            real_sample_used_list.append(sample['REAL_SAMPLE_COUNT'])
            cyber_sample_used_list.append(sample['CYBER_SAMPLE_COUNT'])
            #
            # if sample['REWARD_SUM'] >= 100:
            #     print(sample['REWARD_SUM'])
            #     pass
            # else:
            #     reward_list.append(sample['REWARD_SUM'])
            #     real_sample_used_list.append(sample['REAL_SAMPLE_COUNT'])
            #     cyber_sample_used_list.append(sample['CYBER_SAMPLE_COUNT'])

            count += 1

    min_reward = -100000.0
    pos = 0
    average_reward_list = []
    for i in range(len(reward_list) - eps_size + 1):
        average_reward = sum(reward_list[i: i + eps_size]) / eps_size
        # if average_reward >= 73.0:
        #     print(i, real_sample_used_list[i], cyber_sample_used_list[i], average_reward)
        average_reward_list.append(average_reward)
        # if real_sample_used_list[i] > max_real_sample * 0.5:
        #     if average_reward > min_reward:
        #         min_reward = average_reward
        #         pos = i
        if average_reward > min_reward:
            min_reward = average_reward
            pos = i

    # reward_list.sort(reverse=True)
    # average_reward = sum(reward_list[0:100]) / 100.0
    # print(average_reward)

    return reward_list, real_sample_used_list, cyber_sample_used_list, pos, min_reward, average_reward_list


def print_best_eps_reward(file_path, eps_size):
    reward_list, real_sample_used_list, cyber_sample_used_list, pos, min_reward, average_reward_list = \
        compute_best_eps_reward(test_file=file_path, eps_size=eps_size)

    max_real_sample = max(real_sample_used_list)
    for i in range(len(average_reward_list)):
        print(i, average_reward_list[i])

    print("max reward ", max(reward_list))

    print("100 episode reward %f, at real sample count %d, cyber sample count %d\nmax real sample is %d\n" %
          (min_reward, real_sample_used_list[pos], cyber_sample_used_list[pos], max_real_sample))


def get_sample_count_with_specific_reward(game_dict, game_name, reward_thershold):
    from src.util.plotter import Plotter
    from src.config.config import Config
    from log.logList import LOG_LIST
    key_select_list = ['baseline', 'intel v2 new step']
    all_path_list = []
    all_name = []
    for name, path in game_dict.items():
        if key_select_list is None or name in key_select_list:
            all_path_list.append(path)
            all_name.append(name)
            print(path, name)
    res_dict = {}
    res_dict['reward_value'] = reward_thershold
    res_dict['game_name'] = game_name
    for log_list, name in zip(all_path_list, all_name):
        reward_list, used_sample, _ = Plotter.compute_mean_multi_reward(file_list=Config.load_json(log_list))
        min_used_sample = 1000000000000
        min_reward = -100000000
        for reward, index in zip(reward_list, used_sample):
            if reward >= reward_thershold and index < min_used_sample:
                min_used_sample = index
                min_reward = reward
        res_dict[name + '_used_sample_count'] = min_used_sample
        res_dict[name + '_first_reward'] = min_reward
    with open("/home/dls/CAP/intelligenttrainerframework/log/rewardSampleUsed/" + game_name + '.json', 'w') as f:
        json.dump(res_dict, fp=f, indent=4)


def compute_mean_std_one_exp_and_save(path_list_file, save_dir=None, save_name=None):
    from log.expResult import EXP_RES
    if not save_dir:
        save_dir = EXP_RES
    if not save_name:
        save_name = os.path.basename(path_list_file)
    mean, index, std = Plotter.compute_mean_multi_reward(file_list=Config.load_json(file_path=path_list_file),
                                                         assemble_flag=False)
    res_dict = []
    for mean_i, index_i, std_i in zip(mean, index, std):
        res_dict.append([mean_i, std_i, index_i])
    with open(file=os.path.join(save_dir, save_name), mode='w') as f:
        json.dump(res_dict, f, indent=4)
    print("Saved file at %s %s" % (save_dir, save_name))


def compute_all_res(env_dict):
    for key, val in env_dict.items():
        try:
            print(val)
            compute_mean_std_one_exp_and_save(path_list_file=val)
        except FileNotFoundError:
            print("file not found for %s" % {val})


def compute_one_exp(path_list_file):
    try:
        print(path_list_file)
        compute_mean_std_one_exp_and_save(path_list_file=path_list_file)
    except FileNotFoundError:
        print("file not found for %s" % {path_list_file})


if __name__ == '__main__':
    # compute_mean_std_one_exp_and_save(
    #     path_list_file='/home/dls/CAP/intelligenttrainerframework/log/logList/halfCheetah_baseline_v5base_v5_no_dyna_stepcount=3000.json')
    from test.resDictList import *

    # compute_all_res(mountain_car_continuous_dict)
    compute_one_exp(swimmer_dict['intel_v5_action_split_5'])
