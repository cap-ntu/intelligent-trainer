import numpy as np
import matplotlib

# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import json
import sys
import math as M
from scipy.interpolate import interp1d
from itertools import groupby
import seaborn as sns
import os
import glob
from test.resDictList.reacher import reacher_dict
from test.resDictList.halfCheetah import half_cheetah_dict
from test.resDictList.swimmer import swimmer_dict
from test.resDictList.pendulum import pendulum_dict
from test.resDictList.mountainCarContinuous import mountain_car_continuous_dict
from test.printAllPath import print_all_dir_name
sns.set_style('ticks')

from scipy import stats


def getMostFrquentElement(nums):
    return stats.mode(nums)[0][0]

class Plotter(object):
    markers = ('+', 'x', 'v', 'o', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    color_list = ['b', 'r', 'g', 'm', 'y', 'k', 'cyan', 'plum', 'darkgreen', 'darkorange', 'oldlace', 'chocolate',
                  'purple', 'lightskyblue', 'gray', 'seagreen', 'antiquewhite',
                  'snow', 'darkviolet', 'brown', 'skyblue', 'mediumaquamarine', 'midnightblue', 'darkturquoise',
                  'sienna', 'lightsteelblue', 'gold', 'teal', 'blueviolet', 'mistyrose', 'seashell', 'goldenrod',
                  'forestgreen', 'aquamarine', 'linen', 'deeppink', 'darkslategray', 'mediumseagreen', 'dimgray',
                  'mediumpurple', 'lightgray', 'khaki', 'dodgerblue', 'papayawhip', 'salmon', 'floralwhite',
                  'lightpink', 'gainsboro', 'coral', 'indigo', 'darksalmon', 'royalblue', 'navy', 'orangered',
                  'cadetblue', 'orchid', 'palegreen', 'magenta', 'honeydew', 'darkgray', 'palegoldenrod', 'springgreen',
                  'lawngreen', 'palevioletred', 'olive', 'red', 'lime', 'yellowgreen', 'aliceblue', 'orange',
                  'chartreuse', 'lavender', 'paleturquoise', 'blue', 'azure', 'yellow', 'aqua', 'mediumspringgreen',
                  'cornsilk', 'lightblue', 'steelblue', 'violet', 'sandybrown', 'wheat', 'greenyellow', 'darkred',
                  'mediumslateblue', 'lightseagreen', 'darkblue', 'moccasin', 'lightyellow', 'turquoise', 'tan',
                  'mediumvioletred', 'mediumturquoise', 'limegreen', 'slategray', 'lightslategray', 'mintcream',
                  'darkgreen', 'white', 'mediumorchid', 'firebrick', 'bisque', 'darkcyan', 'ghostwhite', 'powderblue',
                  'tomato', 'lavenderblush', 'darkorchid', 'cornflowerblue', 'plum', 'ivory', 'darkgoldenrod', 'green',
                  'burlywood', 'hotpink', 'cyan', 'silver', 'peru', 'thistle', 'indianred', 'olivedrab',
                  'lightgoldenrodyellow', 'maroon', 'black', 'crimson', 'darkolivegreen', 'lightgreen', 'darkseagreen',
                  'lightcyan', 'saddlebrown', 'deepskyblue', 'slateblue', 'whitesmoke', 'pink', 'darkmagenta',
                  'darkkhaki', 'mediumblue', 'beige', 'blanchedalmond', 'lightsalmon', 'lemonchiffon', 'navajowhite',
                  'darkslateblue', 'lightcoral', 'rosybrown', 'fuchsia', 'peachpuff']

    def __init__(self, log_path):
        self.log_path = log_path + '/loss/'
        self.color_list = Plotter.color_list

        self.markers = Plotter.markers

    def plot_dynamics_env(self):
        test_loss = []
        train_loss = []
        plt.figure(1)
        plt.title('Dynamics Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        with open(file=self.log_path + '/DynamicsEnvMlpModel_train_.log', mode='r') as f:
            train_data = json.load(fp=f)
        with open(file=self.log_path + '/DynamicsEnvMlpModel_test_.log', mode='r') as f:
            test_data = json.load(fp=f)
        for sample in train_data:
            train_loss.append(M.log10(sample['DynamicsEnvMlpModel_LOSS']))
        for sample in test_data:
            test_loss.append(M.log10(sample['DynamicsEnvMlpModel_LOSS']))
        times = len(train_loss) // len(test_loss)

        plt.plot([i * times for i in range(len(test_loss))],
                 test_loss,
                 c='g',
                 label='Test Loss')
        plt.plot([i for i in range(len(train_loss))],
                 train_loss,
                 c='b',
                 label='Train loss')
        plt.legend()
        plt.savefig(self.log_path + '/1.png')

    def plot_target_agent(self):
        test_reward = []
        train_cyber_reward = []
        train_real_count = []
        train_real_reward = []
        plt.figure(2)
        plt.title('Target agent reward')
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        with open(file=self.log_path + 'TargetAgent_train_.log', mode='r') as f:
            train_data = json.load(fp=f)
            for sample in train_data:
                if sample['ENV'] == 'REAL_ENV':
                    train_real_reward.append(sample['REWARD_MEAN'])
                    train_real_count.append(sample['REAL_SAMPLE_COUNT'])
                else:
                    train_cyber_reward.append(sample['REWARD_MEAN'])
        with open(file=self.log_path + 'TargetAgent_test_.log', mode='r') as f:
            test_data = json.load(fp=f)
            for sample in test_data:
                test_reward.append(sample['REWARD_SUM'])

        times = len(train_cyber_reward) // len(train_real_reward)
        if times == 0:
            times = 1
        # plt.plot([i * times for i in range(len(train_real_reward))], train_real_reward, c='g',
        #          label='Train real reward')
        # plt.plot([i for i in range(len(train_cyber_reward))], train_cyber_reward, c='r', label='Train cyber reward')
        plt.plot([i * times for i in range(len(test_reward))], test_reward, c='b', label='test reward')
        plt.legend()
        # plt.show()
        plt.savefig(self.log_path + '/2.png')

    def plot_ddpg_model(self):
        pass
        actor_loss = []
        critic_loss = []
        with open(file=self.log_path + 'DDPGModel_train_.log', mode='r') as f:
            loss = json.load(fp=f)
            for sample in loss:
                actor_loss.append(sample['DDPGModel_ACTOR'])
                # if sample['DDPGModel_ACTOR'] > 0:
                #     actor_loss.append(M.log10(sample['DDPGModel_ACTOR']))
                # else:
                # actor_loss.append(sample['DDPGModel_ACTOR'])
                critic_loss.append(M.log10(sample['DDPGModel_CRITIC']))
        plt.figure(3)

        plt.subplot(2, 1, 1)

        plt.title('Actor Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.plot([i for i in range(len(actor_loss))], actor_loss, c='g', label='Actor loss')

        plt.subplot(2, 1, 2)

        plt.title('Critic Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot([i for i in range(len(critic_loss))], critic_loss, c='r', label='Critic Loss')

        plt.legend()
        plt.savefig(self.log_path + '/3.png')

    @staticmethod
    def plot_multiply_target_agent_reward_no_show(path_list, save_flag=True, title=None, fig_id=4, label=' ',
                                                  save_path=None, assemble_flag=True):

        plt.figure(fig_id)
        if title:
            plt.title(title)
        plt.xlabel('Physic system sample')
        plt.ylabel('Reward Sum')
        for i in range(len(path_list)):
            test_reward = []
            real_env_sample_count_index = []
            if ('assemble' in path_list[i] or 'ensemble' in path_list[i]) and assemble_flag is True:
                for file_i in glob.glob(path_list[i] + '/loss/*BEST_AGENT_TEST_REWARD.json'):
                    file_name = file_i
                    print("Assemble file found %s" % file_name)
                assert file_name is not None
            else:
                file_name = path_list[i] + '/loss/TargetAgent_test_.log'

            with open(file=file_name, mode='r') as f:
                test_data = json.load(fp=f)
                for sample in test_data:
                    test_reward.append(sample['REWARD_SUM'])
                    real_env_sample_count_index.append(sample['REAL_SAMPLE_COUNT'])

            x_keys = []
            y_values = []
            last_key = real_env_sample_count_index[0]
            last_set = []

            for j in range(len(real_env_sample_count_index)):
                if real_env_sample_count_index[j] == last_key:
                    last_set.append(test_reward[j])
                else:
                    x_keys.append(last_key)
                    y_values.append(last_set)
                    last_key = real_env_sample_count_index[j]
                    last_set = [test_reward[j]]
            y_values_mean = [np.mean(y_values[j]) for j in range(len(y_values))]
            plt.plot(x_keys, y_values_mean, c=Plotter.color_list[i], label='Test reward ' + label + str(i))

        plt.legend()
        if save_flag is True:
            for path in path_list:
                plt.savefig(path + '/loss/' + '/compare.png')
        if save_path is not None:
            plt.savefig(save_path)

    @staticmethod
    def plot_any_key_in_log_file(res_dict, res_name, file_name, key, index, scatter_flag=False, save_flag=False,
                                 save_path=None,
                                 fig_id=4, label='', restrict_dict=None, fn=None, path_list=None):
        if not path_list:
            with open(res_dict[res_name], 'r') as f:
                path_list = json.load(f)
        plt.figure(fig_id)
        plt.title("%s_%s_%s" % (res_name, file_name, key))
        plt.xlabel('index')
        plt.ylabel(key)
        for i in range(len(path_list)):
            test_reward = []
            real_env_sample_count_index = []
            with open(file=path_list[i] + '/loss/' + file_name, mode='r') as f:
                test_data = json.load(fp=f)
                for sample in test_data:
                    if fn:
                        if fn(sample) is True:
                            test_reward.append(sample[key])
                            real_env_sample_count_index.append(sample[index])
                    else:
                        if restrict_dict is not None:
                            flag = True
                            for re_key, re_value in restrict_dict.items():
                                if sample[re_key] != re_value:
                                    flag = False
                            if flag is True:
                                test_reward.append(sample[key])
                                real_env_sample_count_index.append(sample[index])
                        else:
                            test_reward.append(sample[key])
                            real_env_sample_count_index.append(sample[index])

            x_keys = []
            y_values = []
            last_key = real_env_sample_count_index[0]
            last_set = []

            for j in range(len(real_env_sample_count_index)):
                if real_env_sample_count_index[j] == last_key:
                    last_set.append(test_reward[j])
                else:
                    x_keys.append(last_key)
                    y_values.append(last_set)
                    last_key = real_env_sample_count_index[j]
                    last_set = [test_reward[j]]
            x_keys.append(last_key)
            y_values.append(last_set)
            y_values_mean = [np.mean(y_values[j]) for j in range(len(y_values))]
            if scatter_flag is True:
                plt.scatter(x_keys, y_values_mean, c=Plotter.color_list[i], label=key + label + str(i),
                            marker=Plotter.markers[i])
            else:
                plt.plot(x_keys, y_values_mean, c=Plotter.color_list[i], label=key + label + str(i),
                         marker=Plotter.markers[i], markevery=10)

        plt.legend()
        if save_flag is True:
            for path in path_list:
                plt.savefig(path + '/loss/' + '/%s_%s.png' % (file_name, key))
        if save_path is not None:
            plt.savefig(save_path)
        # plt.show()

    @staticmethod
    def plot_any_scatter_in_log_file(res_dict, res_name, file_name, key, index, op, scatter_flag=False, save_flag=False,
                                     save_path=None,
                                     fig_id=4, label='', restrict_dict=None):
        with open(res_dict[res_name], 'r') as f:
            path_list = json.load(f)
        plt.figure(fig_id)
        plt.title("%s_%s_%s" % (res_name, file_name, key))
        plt.xlabel('index')
        plt.ylabel(key)
        for i in range(len(path_list)):
            test_reward = []
            real_env_sample_count_index = []
            with open(file=path_list[i] + '/loss/' + file_name, mode='r') as f:
                test_data = json.load(fp=f)
                for sample in test_data:
                    if restrict_dict is not None:
                        flag = True
                        for re_key, re_value in restrict_dict.items():
                            if sample[re_key] != re_value:
                                flag = False
                        if flag is True:
                            test_reward.append(sample[key])
                            real_env_sample_count_index.append(sample[index])
                    else:
                        test_reward.append(sample[key])
                        real_env_sample_count_index.append(sample[index])
            test_reward, real_env_sample_count_index = op(test_reward, real_env_sample_count_index)
            x_keys = []
            y_values = []
            last_key = real_env_sample_count_index[0]
            last_set = []

            for j in range(len(real_env_sample_count_index)):
                if real_env_sample_count_index[j] == last_key:
                    last_set.append(test_reward[j])
                else:
                    x_keys.append(last_key)
                    y_values.append(last_set)
                    last_key = real_env_sample_count_index[j]
                    last_set = [test_reward[j]]
            x_keys.append(last_key)
            y_values.append(last_set)
            y_values_mean = [np.mean(y_values[j]) for j in range(len(y_values))]
            if scatter_flag is True:
                plt.scatter(x_keys, y_values_mean, c=Plotter.color_list[i], label=key + label + str(i),
                            marker=Plotter.markers[i])
            else:
                plt.plot(x_keys, y_values_mean, c=Plotter.color_list[i], label=key + label + str(i),
                         marker=Plotter.markers[i])

        plt.legend()
        if save_flag is True:
            for path in path_list:
                plt.savefig(path + '/loss/' + '/%s_%s.png' % (file_name, key))
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_multiply_target_agent_reward(path_list, fig_id, save_flag=True, title=None, assemble_Flag=True):
        Plotter.plot_multiply_target_agent_reward_no_show(path_list, save_flag, title, fig_id=fig_id,
                                                          assemble_flag=assemble_Flag)


    @staticmethod
    def compute_mean_multi_reward(file_list, assemble_flag=True):

        from test.dataAnalysis import compute_best_eps_reward
        baseline_assemble_reward_list = []
        merged_index_list = []
        for file in file_list:
            merged_reward_list = []
            merged_index_list = []
            file_name = None
            if assemble_flag is True:
                files_ = [file, file + '_FIX_1', file + '_RANDOM_2']
                #### get the rewards data of 3 agents
                reward_lists = []
                real_sample_used_lists = []
                for fkey in files_:
                    file_name = fkey + '/loss/TargetAgent_test_.log'
                    reward_list, real_sample_used_list, _, pos, min_reward, average_reward_list = \
                        compute_best_eps_reward(test_file=file_name)
                    ####merge here such that they are of same length
                    mreward_list = []
                    msampleused_list = []
                    sum_re = 0
                    pre_index = real_sample_used_list[0]
                    sum_count = 0
                    for re, index in zip(reward_list, real_sample_used_list):
                        if index == pre_index:
                            sum_re += re
                            sum_count += 1
                        else:
                            if sum_count == 0:
                                print(sum_count)
                            mreward_list.append(sum_re / sum_count)
                            msampleused_list.append(pre_index)
                            pre_index = index
                            sum_re = re
                            sum_count = 1
                    mreward_list.append(sum_re / sum_count)
                    msampleused_list.append(pre_index)

                    reward_lists.append(mreward_list)
                    real_sample_used_lists.append(msampleused_list)
                if not (len(reward_lists[0]) == len(reward_lists[1]) and len(reward_lists[0]) == len(reward_lists[2])):
                    print("Something wrong")
                assert len(reward_lists[0]) == len(reward_lists[1])
                assert len(reward_lists[0]) == len(reward_lists[2])
                #### get the best player
                best_p_file = fkey + '/loss/GamePlayer_train_.log'
                best_player = []
                sample_index = []
                with open(file=best_p_file, mode='r') as f:
                    test_data = json.load(fp=f)
                    for sample in test_data:
                        best_player.append(sample["BEST_PLAYER"])
                        sample_index.append(sample["REAL_ENV_COUNT"])

                ### accumulate the bets infor for MCar
                # ns = len(best_player)
                # best_player_tmp = []
                # sample_index_tmp = []
                # for i in range(int(len(best_player)/ns)):
                #     best_player_tmp.append(getMostFrquentElement(best_player[i*5: (i+1)*5]))
                #     sample_index_tmp.append(np.max(sample_index[i*5: (i+1)*5]))
                #
                # best_player = best_player_tmp[0:1]
                # sample_index = sample_index_tmp[0:1]

                kindex = 0
                real_sample_used_list = real_sample_used_lists[0]
                reward_list = []
                break_flag = False
                bestp = 0
                for i in range(len(sample_index)):
                    bestp = best_player[i]
                    realc = sample_index[i]
                    while real_sample_used_list[kindex] <= realc:
                        reward_list.append(reward_lists[bestp][kindex])
                        kindex += 1
                        if kindex >= len(real_sample_used_list):
                            break_flag = True
                            break
                    if break_flag:
                        break
                while kindex < len(real_sample_used_list):
                    reward_list.append(reward_lists[bestp][kindex])
                    kindex += 1
                assert len(reward_list) == len(real_sample_used_list)
            else:
                file_name = file + '/loss/TargetAgent_test_.log'
                reward_list, real_sample_used_list, _, pos, min_reward, average_reward_list = \
                    compute_best_eps_reward(test_file=file_name)
            sum_re = 0
            pre_index = real_sample_used_list[0]
            sum_count = 0
            for re, index in zip(reward_list, real_sample_used_list):
                if index == pre_index:
                    sum_re += re
                    sum_count += 1
                else:
                    if sum_count == 0:
                        print(sum_count)
                    merged_reward_list.append(sum_re / sum_count)
                    merged_index_list.append(pre_index)
                    pre_index = index
                    sum_re = re
                    sum_count = 1
            baseline_assemble_reward_list.append(merged_reward_list)
        min_len = len(baseline_assemble_reward_list[0])
        for re_list in baseline_assemble_reward_list:
            min_len = min(min_len, len(re_list))

        for i in range(len(baseline_assemble_reward_list)):
            baseline_assemble_reward_list[i] = baseline_assemble_reward_list[i][0: min_len]
        baseline_assemble_reward_list_std = np.std(np.array(baseline_assemble_reward_list), axis=0)
        baseline_assemble_reward_list = np.mean(np.array(baseline_assemble_reward_list), axis=0)
        return baseline_assemble_reward_list, merged_index_list, baseline_assemble_reward_list_std

    @staticmethod
    def plot_mean_multiply_target_agent_reward(baseline_list, intel_list, save_path=None):
        baseline_reward_list, baseline_index, baseline_std = Plotter.compute_mean_multi_reward(file_list=baseline_list)
        intel_reward_list, intel_index, intel_std = Plotter.compute_mean_multi_reward(file_list=intel_list)
        a = Plotter(log_path='')
        a.plot_fig(fig_num=4, col_id=1, x=baseline_index, y=baseline_reward_list,
                   title='',
                   x_lable='Number of real data samples used', y_label='Reward', label='Baseline Trainer',
                   marker=Plotter.markers[4])
        plt.fill_between(x=baseline_index,
                         y1=baseline_reward_list - baseline_std,
                         y2=baseline_reward_list + baseline_std,
                         alpha=0.3,
                         facecolor=a.color_list[1],
                         linewidth=0)

        a.plot_fig(fig_num=4, col_id=2, x=intel_index, y=intel_reward_list,
                   title='',
                   x_lable='Number of real data samples used', y_label='Reward', label='Intelligent Trainer',
                   marker=Plotter.markers[8])
        plt.fill_between(x=intel_index,
                         y1=intel_reward_list - intel_std,
                         y2=intel_reward_list + intel_std,
                         alpha=0.3,
                         facecolor=a.color_list[2],
                         linewidth=0)
        if save_path is not None:
            plt.savefig(save_path + '/compare.png', bbox_inces='tight')
            plt.savefig(save_path + '/compare.eps', format='eps', bbox_inces='tight')
            plt.savefig(save_path + '/compare.pdf', format='pdf', bbox_inces='tight')

        plt.show()
        pass

    @staticmethod
    def plot_many_target_agent_reward(path_list, name_list, title=' ', assemble_index=[], base_data={},fnum=4):
        i=0
        for i in range(len(path_list)):
            if i in assemble_index:
                assemble_flag = True
            else:
                assemble_flag = False
            baseline_reward_list, baseline_index, baseline_std = Plotter.compute_mean_multi_reward(
                file_list=path_list[i],
                assemble_flag=assemble_flag)

            a = Plotter(log_path='')
            a.plot_fig(fig_num=fnum, col_id=i, x=baseline_index, y=baseline_reward_list,
                       title=title,
                       x_lable='Number of real data samples used', y_label='Reward', label=name_list[i],
                       marker=Plotter.markers[i])
            print(Plotter.markers[i])
            plt.fill_between(x=baseline_index,
                             y1=baseline_reward_list - baseline_std,
                             y2=baseline_reward_list + baseline_std,
                             alpha=0.3,
                             facecolor=Plotter.color_list[i],
                             linewidth=0)
        i = len(path_list)
        if len(base_data) > 0:
            # ['NoCyber', 'Fixed', 'Random', 'DQN', 'DQN-5 actions', 'DQN-larger memory','REINFORCE']: #base_data:
            # ['Ensemble no share memory', 'Ensemble no reference sampling']
            # ['NoCyber', 'Random', 'DQN']
            keys = ['NoCyber', 'DQN', 'Fixed', 'Random', 'DQN-5 actions', 'DQN-larger memory', 'REINFORCE',
                    'DQN-last real reward', 'DQN-sample_count as state']
            perform_c = []
            for key in keys:
                fname = base_data[key]
                with open(fname, 'r') as f:
                    content = json.load(fp=f)
                    raw_data = np.asarray(content)
                    baseline_reward_list = raw_data[:, 0]
                    baseline_std = raw_data[:, 1]
                    baseline_index = raw_data[:, 2]
                    perform_c.append(sum(baseline_reward_list))
                    a = Plotter(log_path='')
                    a.plot_fig(fig_num=fnum, col_id=i, x=baseline_index, y=baseline_reward_list,
                               title=title,
                               x_lable='Number of real data samples used', y_label='Reward', label=key,
                               marker=Plotter.markers[i])
                    print(Plotter.markers[i])
                    plt.fill_between(x=baseline_index,
                                     y1=baseline_reward_list - baseline_std,
                                     y2=baseline_reward_list + baseline_std,
                                     alpha=0.3,
                                     facecolor=Plotter.color_list[i],
                                     linewidth=0)
                i += 1
            print('perform_c=', perform_c)
        # plt.show()

    @staticmethod
    def plot_multiply_target_agent_reward_MEAN(path_list_list, title="IntelTrainer", max_count=10000, legends=[],
                                               assemble_flag=False):
        color_list = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'cyan', 'plum', 'darkgreen']
        plt.figure(4)
        plt.title('Target agent reward')
        plt.xlabel('Physic system sample')
        plt.ylabel('Reward')
        x_new = np.arange(0, 10000)  #####special set for pen, need to be changed for other cases

        for kkk in range(len(path_list_list)):
            y_new_set = []
            path_list = path_list_list[kkk]
            for i in range(len(path_list)):
                test_reward = []
                real_env_sample_count_index = []
                file_name = None
                if assemble_flag is True:
                    for file in glob.glob(path_list[i] + '/loss/*BEST_AGENT_TEST_REWARD.json'):
                        file_name = file
                    assert file_name is not None
                else:
                    file_name = '/loss/TargetAgent_test_.log'

                with open(file=path_list[i] + file_name, mode='r') as f:
                    test_data = json.load(fp=f)
                    for sample in test_data:
                        test_reward.append(sample['REWARD_MEAN'])
                        real_env_sample_count_index.append(sample['REAL_SAMPLE_COUNT'])

                x_keys = []
                y_values = []
                last_key = real_env_sample_count_index[0]
                last_set = []

                for j in range(len(real_env_sample_count_index)):
                    if real_env_sample_count_index[j] == last_key:
                        last_set.append(test_reward[j])
                    else:
                        x_keys.append(last_key)
                        y_values.append(last_set)
                        last_key = real_env_sample_count_index[j]
                        last_set = [test_reward[j]]
                y_values_mean = [np.mean(y_values[j]) for j in range(len(y_values))]

                f_inter__ = interp1d(x_keys, y_values_mean, fill_value="extrapolate")

                y_new = f_inter__(x_new)
                y_new_set.append(y_new)
            y_new_set = np.asarray(y_new_set)
            y_mean = np.mean(y_new_set, 0)
            y_std = np.std(y_new_set, 0)
            print("y_std=", y_std)
            plt.plot(x_new, y_mean, color_list[kkk])
            plt.fill_between(x_new, y_mean - y_std, y_mean + y_std,
                             alpha=0.5, facecolor=color_list[kkk],
                             linewidth=0)
            # plt.errorbar(x_new, y_mean, yerr=y_std)
            # plt.plot(x_new, y_mean)
        plt.legend(legends)
        plt.title(title)
        for path in path_list:
            plt.savefig(path + '/loss/' + '/' + title + '.png')
        plt.show()

    def plot_error(self, p1, p2):
        test1 = []
        test2 = []
        with open(file=p1, mode='r') as f:
            test_data = json.load(fp=f)
            for data in test_data:
                test1.append(data)

        with open(file=p2, mode='r') as f:
            test_data = json.load(fp=f)
            for data in test_data:
                test2.append(data)
        plt.plot(test1, c=self.color_list[1], label='self')
        plt.plot(test2, c=self.color_list[2], label='benchmark')
        plt.legend()

    def _plot(self, x, y):
        pass

    def plot_fig(self, fig_num, col_id, x, y, title, x_lable, y_label, label=' ', marker='*'):
        sns.set_style("darkgrid")
        plt.figure(fig_num, figsize=(6, 5))
        plt.title(title)
        plt.xlabel(x_lable)
        plt.ylabel(y_label)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.tight_layout()

        marker_every = max(int(len(x) / 10), 1)
        if len(np.array(y).shape) > 1:
            new_shape = np.array(y).shape

            res = np.reshape(np.reshape(np.array([y]), newshape=[-1]), newshape=[new_shape[1], new_shape[0]],
                             order='F').tolist()
            res = list(res)
            for i in range(len(res)):
                res_i = res[i]
                plt.subplot(len(res), 1, i + 1)
                plt.title(title + '_' + str(i))
                plt.plot(x, res_i, self.color_list[col_id], label=label + '_' + str(i), marker=marker,
                         markevery=marker_every, markersize=6, linewidth=1)
                # sns.lmplot(x, res_i, self.color_list[col_id], label=label + '_' + str(i), marker=marker, markevery=marker_every, markersize=24)
                col_id += 1
        else:
            plt.plot(x, y, self.color_list[col_id], label=label, marker=marker, markevery=marker_every, markersize=6,
                     linewidth=1)
        plt.legend()

    def log_intel_action(self, path):
        action = []
        obs = []
        val_loss = []
        val_loss_change = []
        contrl_loss = []
        control_loss_change = []
        reward = []
        with open(path, 'r') as f:
            content = json.load(fp=f)
            for sample in content:
                action.append(sample['ACTION'])
                obs.append(sample['OBS'])
                val_loss.append(sample['VALUE_FUNCTION_LOSS'])
                val_loss_change.append(sample['VALUE_FUNCTION_LOSS_CHANGE'])
                contrl_loss.append(sample['CONTROLLER_LOSS_CHANGE'])
                control_loss_change.append(sample['CONTROLLER_LOSS_CHANGE'])
                reward.append(sample['REWARD'])
        return action, obs, val_loss, val_loss_change, contrl_loss, control_loss_change, reward

    def plot_intel_action(self, path):
        action, obs, val_loss, val_loss_change, contrl_loss, control_loss_change, reward = self.log_intel_action(
            path=path)
        self.plot_fig(fig_num=1, col_id=1, x=[i for i in range(len(action))],
                      y=action, title='action', x_lable='', y_label='')

        self.plot_fig(fig_num=2, col_id=2, x=[i for i in range(len(obs))],
                      y=obs, title='obs', x_lable='', y_label='')

        self.plot_fig(fig_num=3, col_id=3, x=[i for i in range(len(val_loss))],
                      y=val_loss, title='val_loss', x_lable='', y_label='')

        self.plot_fig(fig_num=4, col_id=4, x=[i for i in range(len(val_loss_change))],
                      y=val_loss_change, title='val_loss_change', x_lable='', y_label='')

        self.plot_fig(fig_num=5, col_id=5, x=[i for i in range(len(contrl_loss))],
                      y=contrl_loss, title='contrl_loss', x_lable='', y_label='')

        self.plot_fig(fig_num=6, col_id=6, x=[i for i in range(len(control_loss_change))],
                      y=control_loss_change, title='control_loss_change', x_lable='', y_label='')

        self.plot_fig(fig_num=7, col_id=7, x=[i for i in range(len(reward))],
                      y=reward, title='reward', x_lable='', y_label='')

    @staticmethod
    def plot_intel_actions(path_list_list, labels):
        for jjj in range(len(path_list_list)):
            path_list = path_list_list[jjj]
            action_list = []
            for iii in range(len(path_list)):
                path = path_list[iii] + '/loss/TrainerEnv_train_.log'
                action = []
                obs = []
                val_loss = []
                val_loss_change = []
                contrl_loss = []
                control_loss_change = []
                reward = []
                with open(path, 'r') as f:
                    content = json.load(fp=f)
                    for sample in content:
                        action.append(sample['ACTION'])
                        obs.append(sample['OBS'])
                        val_loss.append(sample['VALUE_FUNCTION_LOSS'])
                        val_loss_change.append(sample['VALUE_FUNCTION_LOSS_CHANGE'])
                        contrl_loss.append(sample['CONTROLLER_LOSS_CHANGE'])
                        control_loss_change.append(sample['CONTROLLER_LOSS_CHANGE'])
                        reward.append(sample['REWARD'])
                action = np.asarray(action)
                action_list.append(action)
            mean_action = np.mean(action_list, axis=0)
            mean_action = 1.0 * action_list[0]
            for iii in range(len(action_list) - 1):
                mean_action += action_list[iii + 1]
            mean_action /= len(action_list)
            action_list = np.asarray(action_list)
            x = [i for i in range(len(mean_action))]
            x = np.asarray(x)
            x = x/max(x)
            std_action = np.std(action_list, 0)
            for col in [2]:
                # plt.figure()
                # plt.plot(x, mean_action[:, col])
                Plotter(log_path='').plot_fig(fig_num=col, col_id=jjj, x=x,
                                              y=mean_action[:, col], title='', x_lable='TPE steps (normalized into range [0, 1])',
                                              y_label='Action value', label=labels[jjj], marker = Plotter.markers[jjj])
                plt.ylim(ymax=1.0, ymin=0.0)

                plt.fill_between(x, mean_action[:, col] - std_action[:, col], mean_action[:, col] + std_action[:, col], alpha=0.3)
            # plt.show()


if __name__ == '__main__':
    from log.baselineTestLog import LOG

    path_list = print_all_dir_name(
        father_dir='~/intelligenttrainerpublic/log/intelligentTestLog/Swimmer-v1/',
        contain_char='EnsFinal',
        not_contain_char=['FIX_1', 'RANDOM',])
    all_data = []
    idx = 0
    if 'Pendulum-v0' in path_list[0]:
        idx = 3
    elif 'MountainCarContinuous-v0' in path_list[0]:
        idx = 4
    elif 'Reacher' in path_list[0]:
        idx = 0
    elif 'Swimmer' in path_list[0]:
        idx = 1
    elif 'Half' in path_list[0]:
        idx = 2
    else:
        print("Wrong path")

    #### Example to plot Mean-Std figure
    #### plot ensemble results, leave assemble_index empty to plot single-head results
    Plotter.plot_many_target_agent_reward([path_list], name_list=['Ensemble'], title='', assemble_index=[0])

    #### plot individual (or single head) results
    Plotter.plot_multiply_target_agent_reward(path_list, fig_id=5, save_flag=False, title='Test', assemble_Flag=False)
    plt.show()
