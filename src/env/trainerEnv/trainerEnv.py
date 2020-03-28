from src.env.env import BasicEnv
from src.env.trainerEnv.baselineTrainerEnv import BaselineTrainerEnv
from gym.spaces.box import Box
import numpy as np
from src.config.config import Config
from conf import CONFIG
from conf.key import CONFIG_KEY
import tensorflow as tf
from collections import deque
from scipy.optimize import curve_fit
from statsmodels.tsa.arima_model import AR
import pandas as pd
from src.env.util.trainerEnvStep import TrainerEnvStep


def fit_loss(y_data):
    data = y_data
    model = AR(data)
    res = model.fit()
    ans = res.predict(len(y_data) - 1, len(y_data), dynamic=True)[-1]
    return ans


class TrainerEnv(BaselineTrainerEnv):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/trainerEnvKey.json')

    def __init__(self, config, cyber_env, real_env, target_agent, test_env):
        super(TrainerEnv, self).__init__(config=config,
                                         cyber_env=cyber_env,
                                         real_env=real_env,
                                         target_agent=target_agent,
                                         test_env=test_env)

        high = np.ones([3, ])
        low = 0.2 * high

        self.action_space = Box(low=low,
                                high=high)

        self.observation_space = Box(low=-1e10,
                                     high=1e10,
                                     shape=np.array(self.config.config_dict['STATE_SPACE']))

        self.td = self.config.config_dict['TD']

        self.real_r_his = [-100.0, -100.0, -100.0]
        self.cyber_r_his = [-100.0, -100.0, -100.0]
        self.dyna_err_his = [-100.0, -100.0, -100.0]
        self.status = self.status_key['TRAIN']
        self.stepper = TrainerEnvStep(config=None,
                                      baseline_env_step_type=self.config.config_dict['BASELINE_ENV_STEP_TYPE'],
                                      registred_type=self.config.config_dict['TRAINER_ENV_STEP_TYPE'],
                                      reward_type=self.config.config_dict['REWARD_TYPE'])

    def step(self, action):
        return self.stepper.step(env=self, action=action)


    def _get_obs(self):
        if self.config.config_dict['TRAINER_ENV_STEP_TYPE'] == 'REWARD_DIFFER_REAL_CYBER_MEAN_REWARD_V3':
            return 1.0*self.target_agent._real_env_sample_count / self.target_agent.config.config_dict['MAX_SAMPLE_COUNT']  #[self.real_r_his[-1], self.cyber_r_his[-1]]
        else:
            return 1.0*self.target_agent._real_env_sample_count / self.target_agent.config.config_dict['MAX_SAMPLE_COUNT'] #self.real_r_his[-1]  # re #re #self.target_agent._real_env_sample_count

    def reset(self):
        super().reset()

    def init(self):
        super().init()

    def close(self):
        super().close()

    def get_state(self, env):
        return self._get_obs()
