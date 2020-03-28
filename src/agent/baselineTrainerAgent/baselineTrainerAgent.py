from src.agent.agent import Agent
# import argparse
# import time
# import os
# import logging
# from baselines import logger, bench
# from baselines.common.misc_util import (
#     set_global_seeds,
#     boolean_flag,
# )
# import baselines.ddpg.training as training
# from baselines.ddpg.models import Actor, Critic
# from baselines.ddpg.memory import Memory
# from baselines.ddpg.noise import *
#
# import gym
# import tensorflow as tf
# from mpi4py import MPI
from src.config.config import Config
from conf.key import CONFIG_KEY
from conf import CONFIG
import config as cfg
import numpy as np

class BaselineTrainerAgent(Agent):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/baselineTrainerAgentKey.json')

    def __init__(self, config, model, env):
        super(BaselineTrainerAgent, self).__init__(config=config, model=model, env=env)

    def predict(self, state, *args, **kwargs):
        if self.assigned_action is not None:
            ac = list(self.assigned_action)
            self.assigned_action = None
            state = np.reshape(state, [1, -1])
            re = np.array(self.model.predict(state=state))
            if len(re) > len(ac):
                for i in range(len(ac), len(re)):
                    ac.append(re[i])
            return np.array(ac)
        else:
            ac = np.array(self.model.predict(state=state))
            if 'F1=0' in cfg.config_dict and cfg.config_dict['F1=0'] is True:
                ac[0] = 0.0
            if 'F2=0' in cfg.config_dict and cfg.config_dict['F2=0'] is True:
                ac[1] = 0.0
            return ac

    def init(self):
        self.model.init()
        super().init()

    def store_one_sample(self, state, next_state, action, reward, done, *arg, **kwargs):
        # TODO store the one sample to whatever you want

        self.model.store_one_sample(state=state,
                                    next_state=next_state,
                                    action=action,
                                    reward=reward,
                                    done=done)
        self.log_file_content.append({
            'STATE': np.array(state).tolist(),
            'NEW_STATE': np.array(next_state).tolist(),
            'ACTION': np.array(action).tolist(),
            'REWARD': reward,
            'DONE': done,
            'INDEX': self.log_print_count
        })
        self.log_print_count += 1


if __name__ == '__main__':
    conf = Config(standard_key_list=BaselineTrainerAgent.key_list)
    conf.load_config(path=CONFIG + '/baselineTrainerAgentTestConfig.json')
    a = BaselineTrainerAgent(config=conf, model=None, env=None)
