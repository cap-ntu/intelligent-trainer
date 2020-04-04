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
from concurrent.futures import ThreadPoolExecutor, Future


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
        return
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
        self.step_count = 0
        self.reference_samp = [cfg.config_dict['SAMPLER_PROB'], cfg.config_dict['SAMPLER_PROB']]
        self._thread_executor = ThreadPoolExecutor(max_workers=10)

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
        # TODO add multi-threading here

        for player in self.player_list:
            player.set_seed(seed_new)

        for player in self.player_list:
            player.init()

        self.best_index = -1
        cfg.config_dict['SAMPLER_PROB'] = 0.0

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
        futures = []
        sess = tf.get_default_session()
        g = tf.get_default_graph()
        assert sess is not None, 'tf session is None'
        assert g is not None, 'tf graph is None'

        def call_step(player):
            with sess.as_default():
                with g.as_default():
                    return player.step(step_flag=False)

        for i in range(self.player_count):
            futures.append(self._thread_executor.submit(call_step, (self.player_list[i])))
            # sample = self.player_list[i].step(step_flag=False)
            # sample_list.append(sample)
        for f in futures:
            sample_list.append(f.result())

        self.cumulative_target_agent_real_env_sample_count += \
            self.player_list[0].env.target_agent._real_env_sample_count - pre_target_agent_real_env_sample_count

        if not self.sample_list:
            self.sample_list = sample_list
        else:
            for i in range(len(self.player_list)):
                self.sample_list[i].union(sample_list[i])

        if cfg.config_dict['RANK_REWARD'] is True:
            if self.cumulative_target_agent_real_env_sample_count > cfg.config_dict['RRThre']:
                self.cumulative_target_agent_real_env_sample_count = 0
                for player in self.player_list:
                    player.agent.remain_action_flag = False
                print("raw reward=", [np.sum(self.sample_list[i].reward_set) for i in range(self.player_count)])
                rank_list = np.argsort([np.sum(self.sample_list[i].reward_set) for i in range(self.player_count)])
                reward_factor = (1 - cfg.config_dict['SAMPLER_PROB'])
                reward_factor = reward_factor ** 2
                if cfg.config_dict['SAMPLER_PROB'] > cfg.config_dict['ZeroThre']:
                    reward_factor = 0
                # print("In Player before fake sampling action_iterator=",
                #       self.player_list[0].agent.model.action_iterator)
                print("reward_factor=", reward_factor)
                if reward_factor > 0:
                    for i in range(len(self.sample_list)):
                        self.sample_list[rank_list[i]].reward_set[-1] = reward_factor * float(i)
                        ####
                        if np.isnan(self.sample_list[rank_list[i]].reward_set[-1]):
                            print("Nan observed")
                        # print("In Player before fake sampling action_iterator=",
                        #       self.player_list[0].agent.model.action_iterator)
                        if self.pre_sample_list:
                            self.sample_list[i].action_set[-1][0:2] = self.pre_sample_list[i].action_set[-1][0:2]
                        # print("In Player before fake sampling action_iterator=",
                        #       self.player_list[0].agent.model.action_iterator)
                        self.reward_his[rank_list[i]].append(reward_factor * float(i))
                    print("new rank reward*reward factor=",
                          [self.reward_his[i][-1] for i in range(len(self.sample_list))])
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
                        print("Newly got sample: action={}, reward={}".format(self.sample_list[i].action_set,
                                                                              self.sample_list[i].reward_set))
                        self.pre_sample_list = dp(self.sample_list)
                self.sample_list = None
                self.step_count += 1
            else:
                for player in self.player_list:
                    player.agent.remain_action_flag = True
        else:
            for i in range(self.player_count):
                self.reward_his[i].append(self.sample_list[i].reward_set[-1])
            for i in range(self.player_count):
                self._store_sample_except(sample=self.sample_list[i],
                                          except_list=())
            self.step_count += 1

        NC = cfg.config_dict['NC']
        copy_event = []
        print("self.reward_his[0]=", self.reward_his[0])
        # print("In Player before fake sampling action_iterator=", self.player_list[0].agent.model.action_iterator)
        if len(self.reward_his[0]) >= NC:
            best_index, acc_reward = self._get_best_index_acc_reward()
            self.best_index = best_index
            print("self.best_index=", self.best_index)
            worst_reward = np.min(acc_reward)
            if sum(acc_reward - worst_reward) == 0:
                advan_ratio = 0.5
            else:
                advan_ratio = (acc_reward[best_index] - worst_reward) / sum(acc_reward - worst_reward)

            self.reference_samp[1] = (((advan_ratio - 0.5) / cfg.config_dict['phiRange']) **
                                      cfg.config_dict['POW']) * 1

            self.reference_samp[1] = min(self.reference_samp[1], cfg.config_dict['max_samP'])

            print("advan_ratio=", advan_ratio)
            ####added intel trainer based evaluation
            print("EValution to all actions=")
            self.player_list[0].agent.model.evluate_actions(self.player_list[0].env._get_obs())
            if advan_ratio >= cfg.config_dict['BestThre']:
                for i in range(self.player_count):
                    if i != best_index:
                        if i == 0:
                            print("Weight transferred")
                            self.player_list[i].env.target_agent.model.copy_model(
                                self.player_list[best_index].env.target_agent.model)
                            copy_event.append((best_index, i))
                print("Weight copy finish, best player: %d" % best_index)

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

        print("self.reference_samp[1]=", self.reference_samp[1])

        if self.cumulative_target_agent_real_env_sample_count / 3 < 10 and cfg.config_dict['RRThre'] / 3 < 30:
            self.reference_samp[1] = 0
            self.reference_samp[0] = 0
        if self.step_count < 0:
            cfg.config_dict['SAMPLER_PROB'] = 1.0
        elif self.step_count % cfg.config_dict['SAMPLER_FRE'] == 0:
            cfg.config_dict['SAMPLER_PROB'] = self.reference_samp[0]
        else:
            cfg.config_dict['SAMPLER_PROB'] = self.reference_samp[1]

        # print("In Player before fake sampling action_iterator=", self.player_list[0].agent.model.action_iterator)

        # Limit length of reward his

        if len(self.reward_his[0]) > cfg.config_dict['NC']:
            for i in range(self.player_count):
                self.reward_his[i] = []
        # print("In Player before fake sampling action_iterator=", self.player_list[0].agent.model.action_iterator)
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
        # self.cumulative_target_agent_real_env_sample_count = 0
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
