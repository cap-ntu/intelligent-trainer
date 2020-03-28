import numpy as np
from src.config.config import Config
from conf.key import CONFIG_KEY
from src.util.sampler.sampler import Sampler


class IntelligentSampler(Sampler):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/intelligentSamplerKey.json')

    def __init__(self, cost_fn, config):
        super().__init__(cost_fn, config)
        self.F1 = 0.5
        self.F2 = 0.5
        self.count_new_real_samples = 0

    def sample(self, env, agent, sample_count, store_flag=False, agent_print_log_flag=False, reset_Flag=True):
        path = super().sample(env=env,
                              agent=agent,
                              sample_count=sample_count,
                              store_flag=store_flag,
                              agent_print_log_flag=agent_print_log_flag,
                              reset_Flag=reset_Flag)
        if agent.env_status == agent.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            self.count_new_real_samples = sample_count
        return path

    def reset(self, env, agent, reset_noise=True):
        state = super().reset(env=env, agent=agent, reset_noise=reset_noise)
        vs_list = []
        if agent.sampler.env_status == agent.sampler.config.config_dict['TEST_ENVIRONMENT_STATUS']:
            return state
        print("Entered intel reset")

        if agent.env_status == agent.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            if agent.ref_agent is not None and agent.status == agent.status_key['TRAIN']:
                for kkk_ in range(50):
                    state = env.reset()
                    squo = agent.ref_agent.model.q_value(state=state, step=0)
                    squo = self.F1 * squo + (1 - self.F1) * np.random.rand()
                    vs_list.append(squo)
                    if len(vs_list) > 5 and (np.max(vs_list) - np.min(vs_list)) * 0.999 + np.min(vs_list) < vs_list[-1]:
                        break
            else:
                for kkk_ in range(50):
                    state = env.reset()
                    squo = agent.model.q_value(state=state, step=0)
                    squo = self.F1 * squo + (1 - self.F1) * np.random.rand()
                    vs_list.append(squo)
                    if len(vs_list) > 5 and (np.max(vs_list) - np.min(vs_list)) * 0.999 + np.min(vs_list) < vs_list[-1]:
                        break
        else:
            if np.random.rand() < self.F2:
                idx = np.random.randint(agent.model.real_data_memory.observations0.length)
                state = agent.model.real_data_memory.observations0.get_batch(idx)
                env.set_state(state)
            else:
                state = env.reset()

        return state

    def train(self, state_est_input, state_est_label, dyn_error_est_input, dyn_error_est_label, *args, **kwargs):
        pass

    def set_F(self, F1, F2):
        self.F1 = F1
        self.F2 = F2

    def print_log_queue(self, status):
        self.status = status
