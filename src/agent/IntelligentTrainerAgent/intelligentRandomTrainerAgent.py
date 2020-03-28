from src.agent.agent import Agent
from src.config.config import Config
from conf.key import CONFIG_KEY
from src.agent.baselineTrainerAgent.baselineTrainerAgent import BaselineTrainerAgent
import tensorflow as tf
import numpy as np
import easy_tf_log
from gym.spaces.multi_discrete import MultiDiscrete


class IntelligentRandomTrainerAgent(Agent):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/intelligentRandomTrainerAgentKey.json')

    def __init__(self, config, model, env):
        super(IntelligentRandomTrainerAgent, self).__init__(config=config,
                                                            model=model,
                                                            env=env)
        self.sample_count = 0
        self.sess = tf.get_default_session()
        self.action_space = MultiDiscrete([[0, 1], [0, 1], [0, 1]])

    def predict(self, state, *args, **kwargs):
        res = self.action_space.sample()
        for i in range(3):
            prob = np.random.rand(1)
            if prob <= 0.5:
                res[i] = 1.0
                # res[i] = 0.1
            else:
                res[i] = 0.2
        self.sample_count += 1
        # res = [0.5, 0.9, 0.1]
        return np.array(res)*1.0

    def update(self):
        # TODO finish your own update by using API with self.model
        pass
        # self.model.update()

    def store_one_sample(self, state, next_state, action, reward, done, *arg, **kwargs):
        # TODO store the one sample to whatever you want

        # self.model.store_one_sample(state=state,
        #                             next_state=next_state,
        #                             action=action,
        #                             reward=reward,
        #                             done=done)
        self.log_file_content.append({
            'STATE': np.array(state).tolist(),
            'NEW_STATE': np.array(next_state).tolist(),
            'ACTION': np.array(action).tolist(),
            'REWARD': reward,
            'DONE': done,
            'INDEX': self.log_print_count
        })
        self.log_print_count += 1

    def init(self):
        # TODO init your agent and your model
        # this function will be called at the start of the whole train process
        # self.model.init()
        pass
