import os
import tensorflow as tf
from log.baselineTestLog import LOG
from src.config.config import Config
from conf.key import CONFIG_KEY
import numpy as np
import random
from log.intelligentTestLog import INTEL_LOG
from src.core import Logger
from src.util.sampler.sampler import SamplerData
import config as cfg
import functools
from src.core import Basic
import threading
from threading import Thread
import time
import config as cfg


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


class GamePlayer(Basic):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/gamePlayerKey.json')

    def __init__(self, config, agent, env, basic_list, ep_type=0, log_path=None, log_path_end_with=""):
        super(GamePlayer, self).__init__(config=None)
        self.config = config
        self.agent = agent
        self.env = env
        self.basic_list = []
        for basic in basic_list:
            if basic is not None:
                self.basic_list.append(basic)
        self.basic_list.append(self)
        if ep_type == 1:
            log = INTEL_LOG
        else:
            log = LOG
        self.logger = Logger(prefix=self.config.config_dict['GAME_NAME'],
                             log=log,
                             log_path=log_path,
                             log_path_end=log_path_end_with)

        self.step_count = 0
        self.config.config_dict['START_TIME'] = int(time.time())

    @property
    def real_env_sample_count(self):
        return self.env.target_agent._real_env_sample_count

    def set_seed(self, seed=None):
        if seed is None:
            seed = int(self.config.config_dict['SEED'])
        else:
            self.config.config_dict['SEED'] = seed
        np.random.seed(seed)
        random.seed(seed)

    def save_config(self):
        if self.config.config_dict['SAVE_CONFIG_FILE_FLAG'] == 1:
            for basic in self.basic_list:
                if basic.config is not None:
                    basic.config.save_config(path=self.logger.config_file_log_dir,
                                             name=basic.name + '.json')
            self.config.save_config(path=self.logger.config_file_log_dir,
                                    name='GamePlayer.json')
            # Config.save_to_json(dict=cfg.config_dict,
            #                     file_name='expConfig.json',
            #                     path=self.logger.config_file_log_dir)

    def init(self):
        self.agent.init()
        self.env.init()

    def step(self, step_flag=True):
        trainer_data = self.agent.sample(env=self.env,
                                         sample_count=1,
                                         store_flag=step_flag,
                                         agent_print_log_flag=True)
        self.step_count += 1
        if self.step_count % 1 == 0 and self.step_count > 0:
            self.agent.update()
        return trainer_data

    def play(self, seed_new=None):
        self.set_seed(seed_new)
        self.init()

        info_set = []

        # TODO modify here to control the whole training process
        for i in range(self.config.config_dict['EPOCH']):
            for j in range(self.config.config_dict['STEP']):
                print("\nEPOCH %d, STEP %d" % (i, j))
                trainer_data = self.step()
                # info_set[0].append(self.agent.sampler.info[0])
                # info_set[1].append(self.agent.sampler.info[1])
                # info_set[2].append(self.agent.sampler.info[2])
                info_set.append(trainer_data)
                print("self.config.config_dict['MAX_REAL_ENV_SAMPLE']=", self.config.config_dict['MAX_REAL_ENV_SAMPLE'])
                print("self.env.target_agent._real_env_sample_count=", self.real_env_sample_count)
                if self.real_env_sample_count > self.config.config_dict['MAX_REAL_ENV_SAMPLE']:
                    break
            if self.real_env_sample_count > self.config.config_dict['MAX_REAL_ENV_SAMPLE']:
                break
        # END
        self.config.config_dict['END_TIME'] = int(time.time())
        self.save_config()

        return info_set

    def print_log_to_file(self):
        for basic in self.basic_list:
            if 'LOG_FLAG' in basic.config.config_dict and basic.config.config_dict['LOG_FLAG'] == 1:
                basic.status = basic.status_key['TRAIN']
                self.logger.out_to_file(file_path=os.path.join(self.logger.loss_file_log_dir, basic.log_file),
                                        content=basic.log_file_content)
                basic.status = basic.status_key['TEST']
                self.logger.out_to_file(file_path=os.path.join(self.logger.loss_file_log_dir, basic.log_file),
                                        content=basic.log_file_content)

    def save_all_model(self):
        from src.model.tensorflowBasedModel import TensorflowBasedModel
        for basic in self.basic_list:
            if isinstance(basic, TensorflowBasedModel):
                basic.save_model(path=self.logger.model_file_log_dir, global_step=1)

    def load_all_model(self):
        from src.model.tensorflowBasedModel import TensorflowBasedModel
        for basic in self.basic_list:
            if isinstance(basic, TensorflowBasedModel):
                basic.load_model(path=self.logger.model_file_log_dir, global_step=1)

    def _security_check(self):
        pass


class AssembleGamePlayer(object):
    def __init__(self, intel_player, ref_player_list):
        self.main_player = intel_player
        self.reference_players = ref_player_list
        pass

    @property
    def real_env_sample_count(self):
        return self.main_player.env.target_agent._real_env_sample_count

    def play(self, seed_new=None):
        self.main_player.set_seed(seed_new)
        for player in self.reference_players:
            player.set_seed(seed_new)

        self.main_player.init()
        for player in self.reference_players:
            player.init()

        #####combine memory first before each step.

        for i in range(self.main_player.config.config_dict['EPOCH']):
            for j in range(self.main_player.config.config_dict['STEP']):
                print("\nEPOCH %d, STEP %d" % (i, j))
                self.main_player.step()
                for player in self.reference_players:
                    player.step()
                print("self.config.config_dict['MAX_REAL_ENV_SAMPLE']=",
                      self.main_player.config.config_dict['MAX_REAL_ENV_SAMPLE'])
                print("self.env.target_agent._real_env_sample_count=",
                      self.real_env_sample_count)
                if self.real_env_sample_count + self.main_player.env.config.config_dict[
                    'SAMPLE_COUNT_PER_STEP'] > self.main_player.config.config_dict[
                    'MAX_REAL_ENV_SAMPLE'] * self.main_player.config.config_dict['REAL_ENV_SAMPLE_TRAIN_RATION']:
                    break
            if self.real_env_sample_count + self.main_player.env.config.config_dict[
                'SAMPLE_COUNT_PER_STEP'] > self.main_player.config.config_dict[
                'MAX_REAL_ENV_SAMPLE'] * self.main_player.config.config_dict['REAL_ENV_SAMPLE_TRAIN_RATION']:
                break

        self.main_player.config.config_dict['END_TIME'] = int(time.time())
        self.main_player.save_config()

        for player in self.reference_players:
            player.config.config_dict['END_TIME'] = int(time.time())
            player.save_config()

        self.print_log_to_file()
        self.save_all_model()
        self.final_test_process()

    def print_log_to_file(self):
        self.main_player.print_log_to_file()
        for player in self.reference_players:
            player.print_log_to_file()

    def save_all_model(self):
        self.main_player.save_all_model()
        for player in self.reference_players:
            player.save_all_model()

    def final_test_process(self):
        test_sample_real_env_count = self.main_player.config.config_dict['MAX_REAL_ENV_SAMPLE'] - \
                                     self.real_env_sample_count

        target_agent_count = 1 + len(self.reference_players)
        test_sample_per_target_agent = int(test_sample_real_env_count // target_agent_count)

        best_reward = -1000000.0
        best_player = None
        print('Real env used %d Test sample for each player %d\n' % (
            self.real_env_sample_count, test_sample_per_target_agent))
        reward = self._test_agent(env=self.main_player.env.test_env,
                                  target_agent=self.main_player.env.target_agent,
                                  test_sample_per_agent=test_sample_per_target_agent
                                  )
        print("Mean Reward of %s is %f" % (self.main_player.env.target_agent.config.config_dict['NAME'], reward))
        if reward > best_reward:
            best_reward = reward
            best_player = self.main_player
        for player in self.reference_players:
            reward = self._test_agent(env=player.env.test_env,
                                      target_agent=player.env.target_agent,
                                      test_sample_per_agent=test_sample_per_target_agent
                                      )
            print("Mean Reward of %s is %f" % (player.env.target_agent.config.config_dict['NAME'], reward))

            if reward > best_reward:
                best_reward = reward
                best_player = player

        print("Best player is %s its reward is %f" % (
            best_player.env.target_agent.config.config_dict['NAME'], best_reward))

        best_player.env.target_agent.status = best_player.env.target_agent.status_key['TEST']
        best_reward_log_content = best_player.env.target_agent.log_file_content
        file_name = best_player.env.target_agent.config.config_dict['NAME'] + '_BEST_AGENT_TEST_REWARD.json'

        self.main_player.logger.out_to_file(
            file_path=os.path.join(self.main_player.logger.loss_file_log_dir, file_name),
            content=best_reward_log_content)
        for player in self.reference_players:
            player.logger.out_to_file(
                file_path=os.path.join(player.logger.loss_file_log_dir, file_name),
                content=best_reward_log_content)

    def _test_agent(self, env, target_agent, test_sample_per_agent):
        target_agent.env_status = target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS']
        target_agent.status = target_agent.status_key['TEST']
        test_data = target_agent.sample(env=env,
                                        sample_count=test_sample_per_agent,
                                        store_flag=False,
                                        agent_print_log_flag=True)
        mean_reward = float(np.mean(test_data.reward_set))
        return mean_reward


class RandomEnsemblePlayer(Basic):
    def __init__(self, player_list, intel_trainer_index=None, fakeSamplers=None):
        super(RandomEnsemblePlayer, self).__init__(config=None)
        self.player_list = player_list
        self.player_count = len(player_list)
        self.total_real_env_sample = self.player_list[0].config.config_dict['MAX_REAL_ENV_SAMPLE']
        self.intel_trainer_index = intel_trainer_index
        self.fakeSamplers = fakeSamplers
        self.sample_list = None
        self.reward_his = [[] for i in range(self.player_count)]
        self._best_index = 1
        self.cumulative_target_agent_real_env_sample_count = 0
        self.pre_sample_list = None

    @property
    def best_index(self):
        return self._best_index

    @best_index.setter
    def best_index(self, new_val):
        self._best_index = new_val
        if 'REF_AGENT' in cfg.config_dict and cfg.config_dict['REF_AGENT'] is True:
            for i in range(len(self.player_list)):
                if self.best_index == -1:
                    self.player_list[i].env.target_agent.ref_agent = self.player_list[i].env.target_agent
                else:
                    self.player_list[i].env.target_agent.ref_agent = self.player_list[self.best_index].env.target_agent

    @property
    def real_env_sample_count(self):
        return self.player_list[0].env.target_agent._real_env_sample_count

    def play(self, seed_new=None):
        for player in self.player_list:
            player.set_seed(seed_new)

        for player in self.player_list:
            player.init()

        self.best_index = 1

        for i in range(self.player_list[0].config.config_dict['EPOCH']):
            for j in range(self.player_list[0].config.config_dict['STEP']):
                print("\nEPOCH %d, STEP %d" % (i, j))
                self.step()
                print("MAX REAL ENV SAMPLE %d" % self.total_real_env_sample)
                print("self.env.target_agent._real_env_sample_count=",
                      self.real_env_sample_count)
                if self.real_env_sample_count > self.total_real_env_sample:
                    break
            if self.real_env_sample_count > self.total_real_env_sample:
                break

        for player in self.player_list:
            player.config.config_dict['END_TIME'] = int(time.time())
            player.save_config()

        self.print_log_to_file()
        self.save_all_model()

    def print_log_to_file(self):
        for player in self.player_list:
            player.print_log_to_file()

    def save_all_model(self):
        for player in self.player_list:
            player.save_all_model()

    def step(self):
        pre_target_agent_real_env_sample_count = self.player_list[0].env.target_agent._real_env_sample_count
        """Share the real memory with fakeSamplers"""
        for i in range(len(self.fakeSamplers)):
            for j in range(len(self.fakeSamplers)):
                if j != i:
                    if self.fakeSamplers[i].reference_trainer_env.target_agent.model.enough_data(
                            sample_count=self.player_list[j].env.config.config_dict['SAMPLE_COUNT_PER_STEP'],
                            env_status=self.player_list[j].env.target_agent.config.config_dict[
                                'REAL_ENVIRONMENT_STATUS']) is True:
                        print("Enough data for %d %d" % (i, j))
                        self.player_list[j].env.target_agent.status = self.player_list[j].env.status_key['TRAIN']
                        self.player_list[j].env.target_agent.env_status = \
                            self.player_list[j].env.target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS']
                        self.fakeSamplers[i].sample(env=self.player_list[j].env.real_env,
                                                    agent=self.player_list[j].env.target_agent,
                                                    sample_count=self.player_list[j].env.config.config_dict[
                                                        'SAMPLE_COUNT_PER_STEP'],
                                                    store_flag=True, agent_print_log_flag=False, reset_Flag=True)
                    else:
                        print('No enough data')
        sample_list = []

        from copy import deepcopy as dp
        for i in range(self.player_count):
            sample = self.player_list[i].step(step_flag=False)
            sample_list.append(sample)
        self.cumulative_target_agent_real_env_sample_count += self.player_list[
                                                                  0].env.target_agent._real_env_sample_count - pre_target_agent_real_env_sample_count

        if not self.sample_list:
            self.sample_list = sample_list
        else:
            for i in range(len(self.player_list)):
                self.sample_list[i].union(sample_list[i])

        if cfg.config_dict['RANK_REWARD'] is True:
            if self.cumulative_target_agent_real_env_sample_count > 30:
                self.cumulative_target_agent_real_env_sample_count = 0
                for player in self.player_list:
                    player.agent.remain_action_flag = False
                rank_list = np.argsort([np.sum(self.sample_list[i].reward_set) for i in range(self.player_count)])
                for i in range(len(self.sample_list)):
                    self.sample_list[rank_list[i]].reward_set[-1] = float(i)
                    if self.pre_sample_list:
                        self.sample_list[i].action_set[-1][0:2] = self.pre_sample_list[i].action_set[-1][0:2]
                    self.reward_his[rank_list[i]].append(float(i))

                for i in range(len(self.player_list)):
                    self.sample_list[i].state_set = self.sample_list[i].state_set[-1:]
                    self.sample_list[i].action_set = self.sample_list[i].action_set[-1:]
                    self.sample_list[i].reward_set = self.sample_list[i].reward_set[-1:]
                    self.sample_list[i].done_set = self.sample_list[i].done_set[-1:]
                    self.sample_list[i].new_state_set = self.sample_list[i].new_state_set[-1:]
                    self.sample_list[i].step_count_per_episode = 1
                    self.sample_list[i].cumulative_reward = self.sample_list[i].reward_set[-1]
                for i in range(self.player_count):
                    self._store_sample_except(sample=self.sample_list[i],
                                              except_list=())
                    self.pre_sample_list = dp(self.sample_list)
                self.sample_list = None
        else:
            for i in range(self.player_count):
                self.reward_his[i].append(self.sample_list[i][0].reward_set[-1])
            for i in range(self.player_count):
                self._store_sample_except(sample=self.sample_list[i],
                                          except_list=())

        NC = cfg.config_dict['NC']
        clearFlag = False
        copy_event = []
        if len(self.reward_his[0]) >= NC:
            # every NC reward a test
            best_index, acc_reward = self._get_best_index_acc_reward()
            self.best_index = best_index

            advan_ratio = acc_reward[best_index] / sum(acc_reward)

            cfg.config_dict['SAMPLER_PROB'] = (((advan_ratio - 0.5) / (cfg.config_dict['BestThre'] - 0.5)) **
                                               cfg.config_dict['POW']) * cfg.config_dict['max_samP']

            cfg.config_dict['SAMPLER_PROB'] = min(cfg.config_dict['SAMPLER_PROB'], cfg.config_dict['max_samP'])

            # if advan_ratio >= cfg.config_dict['BestThre']:
            #     cfg.config_dict['COPY_PARTLY'] = True
            # else:
            #     cfg.config_dict['COPY_PARTLY'] =False

            if advan_ratio >= cfg.config_dict['BestThre']:
                for i in range(self.player_count):
                    if i != best_index:
                        # TODO PARTLY COPY FLAG
                        if 'COPY_PARTLY' in cfg.config_dict and cfg.config_dict['COPY_PARTLY'] is True and i == 1:
                            continue
                        else:
                            self.player_list[i].env.target_agent.model.copy_model(
                                self.player_list[best_index].env.target_agent.model)
                            copy_event.append((best_index, i))
                print("Weigh copy finish, best player: %d" % best_index)

                clearFlag = True

            for player in self.player_list:
                player.log_file_content.append({
                    'BEST_REWARD': sum(self.reward_his[best_index]),
                    'BEST_PLAYER': best_index,
                    'REAL_ENV_COUNT': self.real_env_sample_count,
                    'INDEX': self.log_print_count,
                    'COPY_EVENT': copy_event
                })
                player.log_file_count = player.log_print_count + 1

            self.log_print_count = self.log_print_count + 1

        if clearFlag is True:
            # clear old sample reward

            for i in range(self.player_count):
                self.reward_his[i] = []

        # Limit length of reward his

        while len(self.reward_his[0]) > cfg.config_dict['MAX_RE_LEN']:
            for i in range(self.player_count):
                self.reward_his[i].pop(0)

        if 'STE_V3_TEST_MOVE_OUT' in cfg.config_dict and cfg.config_dict['STE_V3_TEST_MOVE_OUT'] is True:
            self.test()

    def test(self):
        for player in self.player_list:
            player.env.stepper.baseline_step.test(env=player.env)

    def _store_sample_except(self, sample, except_list):
        for j in range(self.player_count):
            if j not in except_list:
                self.player_list[j].agent.store_one_sample(state=sample.state_set[-1],
                                                           action=sample.action_set[-1],
                                                           reward=sample.reward_set[-1],
                                                           done=sample.done_set[-1],
                                                           next_state=sample.new_state_set[-1])

    @staticmethod
    def _rank_reward_func(sample_list):
        sample_list.sort(key=lambda x: x[0].reward_set[-1])
        return sample_list

    @staticmethod
    def discount_sum_reward(re_list, discount):
        assert 1.0 >= discount >= 0.0
        sum = 0.
        dis = 1.
        for i in range(len(re_list)):
            sum += re_list[len(re_list) - i - 1] * dis
            dis = dis * discount
        return sum

    @staticmethod
    def linear_discount_sum_reward(re_list):
        sum = 0.
        if len(re_list) < 2:
            return np.sum(re_list)

        step = (1. - cfg.config_dict['LINEAR_DISCOUNT']) / (len(re_list) - 1)
        for i in range(len(re_list)):
            sum += re_list[i] * (cfg.config_dict['LINEAR_DISCOUNT'] + step * i)
        return sum

    def _get_best_index_acc_reward(self):
        self.cumulative_target_agent_real_env_sample_count = 0
        assert int('DISCOUNT' in cfg.config_dict) + int('LINEAR_DISCOUNT' in cfg.config_dict) < 2
        if 'DISCOUNT' in cfg.config_dict:
            best_index = max(range(self.player_count), key=lambda k: self.discount_sum_reward(self.reward_his[k],
                                                                                              discount=
                                                                                              cfg.config_dict[
                                                                                                  'DISCOUNT']))
            acc_reward = [self.discount_sum_reward(self.reward_his[k], discount=cfg.config_dict['DISCOUNT']) for k
                          in range(self.player_count)]
        elif 'LINEAR_DISCOUNT' in cfg.config_dict and cfg.config_dict['LINEAR_DISCOUNT'] >= 0.0:
            best_index = max(range(self.player_count),
                             key=lambda k: self.linear_discount_sum_reward(self.reward_his[k]))
            acc_reward = [self.linear_discount_sum_reward(self.reward_his[k]) for k
                          in range(self.player_count)]
        else:
            best_index = max(range(self.player_count), key=lambda k: sum(self.reward_his[k]))
            acc_reward = [sum(self.reward_his[k]) for k in range(self.player_count)]
        acc_reward = np.asarray(acc_reward)
        acc_reward -= np.min(acc_reward)
        return best_index, acc_reward


if __name__ == '__main__':
    a = SamplerData()
    b = SamplerData()
    c = SamplerData()
    a.append(state=0, action=0, reward=0, done=True, new_state=0.)
    b.append(state=0, action=0, reward=1., done=True, new_state=0.)
    c.append(state=0, action=0, reward=2., done=True, new_state=0.)
    res = [a, b, c]
    res_ = [(res[i], i) for i in range(3)]
    res = RandomEnsemblePlayer._rank_reward_func(sample_list=res_)
    pass
