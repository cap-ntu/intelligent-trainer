from src.model.model import Model
from src.config.config import Config
from conf.key import CONFIG_KEY
import numpy as np


class FixedOutputModel(Model):
    # key_list = Config.load_json(file_path=CONFIG_KEY + '/fixedOutputModelKey.json')

    def __init__(self, config):
        super(FixedOutputModel, self).__init__(config)

    def predict(self, sess=None, state=None):
        action = [0 for _ in range(self.config.config_dict['ACTION_SPACE'][0])]
        action[0] = 0.6
        action[1] = 0.6
        action[2] = 0.6

        return np.array(action)

    def reset(self):
        pass

    def update(self, *args, **kwargs):
        pass

    def print_log_queue(self, status):
        pass


if __name__ == '__main__':
    from conf import CONFIG

    conf = Config(standard_key_list=FixedOutputModel.key_list)
    conf.load_config(path=CONFIG + '/baselineTrainerModelTestConfig.json')
    a = FixedOutputModel(config=conf)
    print(a.predict(state=1))
