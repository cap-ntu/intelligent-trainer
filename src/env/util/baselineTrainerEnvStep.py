from src.env.util.step import Step
import tensorflow as tf
import numpy as np
import config as cfg


class BaselineTrainerEnvStep(Step):

    def __init__(self, config, registred_type):
        super().__init__(config)
        if registred_type == 'STEP_V3':
            self.step = self.trainer_env_step_v3
        elif registred_type == 'STEP_V4':
            self.step = self.trainer_env_step_v4
        elif registred_type == 'STEP_V3_CONTROL_DYNA_HORIZON':
            self.step = self.trainer_env_step_v3_control_dyna_horizon
        else:
            raise IndexError('Not support %s step function type' % registred_type)

    @staticmethod
    def trainer_env_step_v4(env, action):
        from src.env.trainerEnv.baselineTrainerEnv import BaselineTrainerEnv
        assert isinstance(env, BaselineTrainerEnv)

        env.step_count += 1

        F1 = action[0]
        prob_sample_on_real = action[1]
        prob_train_on_real = action[2]
        train_in_cyber_flag = False

        print("\nTrain for target agent from real env----------------------")
        env.target_agent.status = env.status_key['TRAIN']

        t_r = int(max(env.config.config_dict['TARGET_AGENT_TRAIN_ITERATION'], 1))
        t_c = int((1 - prob_train_on_real) * t_r / prob_train_on_real)

        env.target_agent.env_status = env.target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS']

        for i in range(t_r):
            env.target_agent.train()
            env.target_agent.print_log_queue(status=env.status_key['TRAIN'])

        env.target_agent.sampler.set_F(F1=F1, F2=0.0)

        print("\n Real Env used count %d" % env.target_agent._real_env_sample_count)

        print("\nSample for target agent----------------------")
        print("\nSample for target agent in real env 1---------------------")

        K_r = env.config.config_dict['SAMPLE_COUNT_PER_STEP']
        K_r = max(1, K_r // 2)

        from src.util.sampler.fakeSampler import FakeSampler
        from src.util.sampler.fakeIntelligentSampler import FakeIntelligentSampler
        if isinstance(env.target_agent.sampler, (FakeSampler, FakeIntelligentSampler)):
            sample_count = K_r
            sample_step = 1
        else:
            sample_count = 1
            sample_step = K_r
        env._sample_from_real_env(sample_count=sample_count, sample_step=sample_step)

        print("\nSample for target agent in cyber env ---------------------")

        K_c = int(
            env.config.config_dict['SAMPLE_COUNT_PER_STEP'] / prob_sample_on_real * (1.0 - prob_sample_on_real))
        if K_c < 1:
            K_c = 1

        if isinstance(env.target_agent.sampler, (FakeSampler, FakeIntelligentSampler)):
            sample_count = K_c
            sample_step = 1
        else:
            sample_count = 1
            sample_step = K_c

        env._sample_from_cyber_env(sample_step=sample_step, sample_count=sample_count)

        print("\nTrain for target agent from cyber env----------------------")
        env.target_agent.status = env.status_key['TRAIN']
        env.target_agent.env_status = env.target_agent.config.config_dict['CYBER_ENVIRONMENT_STATUS']
        for i in range(t_c):
            res = env.target_agent.train()
            if res is not None:
                train_in_cyber_flag = True
            env.target_agent.print_log_queue(status=env.status_key['TRAIN'])

        print("\nSample for target agent in real env 2---------------------")

        if isinstance(env.target_agent.sampler, (FakeSampler, FakeIntelligentSampler)):
            sample_count = K_r
            sample_step = 1
        else:
            sample_count = 1
            sample_step = K_r
        env._sample_from_real_env(sample_count=sample_count, sample_step=sample_step)

        env.sample_count += env.config.config_dict['SAMPLE_COUNT_PER_STEP']

        final_step_dynamics_train_loss = -1

        print("\nTrain for dynamics env----------------------")

        env.cyber_env.status = env.status_key['TRAIN']

        for i in range(env.config.config_dict['DYNAMICS_TRAIN_ITERATION']):
            data = env.real_env_sample_memory.sample(
                batch_size=env.cyber_env.model.config.config_dict['BATCH_SIZE'])

            final_step_dynamics_train_loss = env.cyber_env.fit(state_set=data['obs0'],
                                                               action_set=data['action'],
                                                               delta_state_label_set=data['delta'],
                                                               sess=tf.get_default_session())

            env.cyber_env.print_log_queue(env.status_key['TRAIN'])
        env.dyna_error_dequeu.append(final_step_dynamics_train_loss)

        progress_bar = np.floor(
            1.0 * env.target_agent._real_env_sample_count / env.config.config_dict['TEST_FRIQUENCY_SAMPLE'])
        if progress_bar > env.last_test:
            env.last_test = progress_bar
            print("\nTest for dynamics env----------------------")
            env.target_agent.status = env.status_key['TEST']
            env.target_agent.env_status = env.target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS']
            env.cyber_env.status = env.status_key['TEST']

            sample_data = env.target_agent.sample(env=env.test_env,
                                                  sample_count=1000,
                                                  store_flag=False,
                                                  agent_print_log_flag=False)

            env.cyber_env.test(state_set=sample_data.state_set,
                               action_set=sample_data.action_set,
                               delta_state_label_set=np.array(sample_data.new_state_set) - np.array(
                                   sample_data.state_set),
                               sess=tf.get_default_session())
            env.cyber_env.print_log_queue(status=env.status_key['TEST'])

            print("\nTest for target agent by real cost function----------------------")
            env.target_agent.status = env.status_key['TEST']
            env.target_agent.env_status = env.target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS']

            assert env.target_agent.sampler.env_status == \
                   env.target_agent.sampler.config.config_dict['TEST_ENVIRONMENT_STATUS']

            sample_data = env.target_agent.sample(env=env.test_env,
                                                  sample_count=env.config.config_dict['TARGET_AGENT_TEST_SAMPLE'],
                                                  store_flag=False,
                                                  agent_print_log_flag=True)

        obs = 0.0
        reward = np.sign(env.target_agent_real_env_reward_deque[-1] - env.target_agent_real_env_reward_deque[-2])

        done = False
        info = [env.target_agent_real_env_reward_deque[-1], env.target_agent_real_env_reward_deque[-2],
                train_in_cyber_flag]
        print("EEE env.target_agent._real_env_sample_count=", env.target_agent._real_env_sample_count)
        return obs, reward, done, info

    @staticmethod
    def trainer_env_step_v3(env, action):
        from src.env.trainerEnv.baselineTrainerEnv import BaselineTrainerEnv
        assert isinstance(env, BaselineTrainerEnv)
        env.step_count += 1

        F1 = action[0]
        F2 = action[1]
        prob_sample_on_real = action[2]
        prob_train_on_real = action[2]
        train_in_cyber_flag = False

        print("\nTrain for target agent from real env----------------------")
        env.target_agent.status = env.status_key['TRAIN']

        t_r = int(max(env.config.config_dict['TARGET_AGENT_TRAIN_ITERATION'], 1))
        t_c = int((1 - prob_train_on_real) * t_r / prob_train_on_real)

        total_train = int(t_r + t_c)

        for i in range(total_train):
            prob = np.random.rand()
            if prob <= prob_train_on_real:
                env.target_agent.env_status = env.target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS']
                res_dict = env.target_agent.train()
            else:
                env.target_agent.env_status = env.target_agent.config.config_dict['CYBER_ENVIRONMENT_STATUS']
                res_dict = env.target_agent.train()
                if res_dict is not None:
                    train_in_cyber_flag = True

            env.target_agent.print_log_queue(status=env.status_key['TRAIN'])

        env.target_agent.sampler.set_F(F1=F1, F2=F2)

        print("\n Real Env used count %d" % env.target_agent._real_env_sample_count)

        print("\nSample for target agent----------------------")

        env.target_agent.status = env.target_agent.status_key['TRAIN']
        env.target_agent.env_status = env.target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS']

        real_reward_data_this_step = []
        K_r = env.config.config_dict['SAMPLE_COUNT_PER_STEP']

        from src.util.sampler.fakeSampler import FakeSampler
        from src.util.sampler.fakeIntelligentSampler import FakeIntelligentSampler
        if isinstance(env.target_agent.sampler, (FakeSampler, FakeIntelligentSampler)):
            sample_count = K_r
            sample_step = 1
        else:
            sample_count = 1
            sample_step = K_r

        for i in range(sample_step):
            sample_data = env.target_agent.sample(env=env.real_env,
                                                  sample_count=sample_count,
                                                  store_flag=True,
                                                  agent_print_log_flag=True)

            for j in range(len(sample_data.state_set)):
                real_reward_data_this_step.append(sample_data.reward_set[j])

                data_dict = {
                    'obs0': sample_data.state_set[j],
                    'obs1': sample_data.new_state_set[j],
                    'action': sample_data.action_set[j],
                    'reward': sample_data.reward_set[j],
                    'terminal1': sample_data.done_set[j],
                    'delta_state': sample_data.new_state_set[j] - sample_data.state_set[j]
                }
                env.real_env_sample_memory.append(data_dict)
            env.cyber_env.model.update_mean_var(state_input=np.array(sample_data.state_set),
                                                action_input=np.array(sample_data.action_set),
                                                delta_state_label=np.array(sample_data.new_state_set) -
                                                                  np.array(sample_data.state_set))
        env.target_agent_real_env_reward_deque.append(np.mean(real_reward_data_this_step))

        cyber_reward_data_this_step = []
        K_c = int(
            env.config.config_dict['SAMPLE_COUNT_PER_STEP'] / prob_sample_on_real * (1.0 - prob_sample_on_real))
        if K_c < 1:
            K_c = 1

        if isinstance(env.target_agent.sampler, (FakeSampler, FakeIntelligentSampler)):
            sample_count = K_c
            sample_step = 1
        else:
            sample_count = 1
            sample_step = K_c

        env.target_agent.env_status = env.target_agent.config.config_dict['CYBER_ENVIRONMENT_STATUS']
        env.target_agent.status = env.target_agent.status_key['TRAIN']
        # env.target_agent.log_queue.queue.clear()
        for i in range(sample_step):
            sample_data = env.target_agent.sample(env=env.cyber_env,
                                                  sample_count=sample_count,
                                                  store_flag=True,
                                                  agent_print_log_flag=True)
            for j in range(len(sample_data.state_set)):
                cyber_reward_data_this_step.append(sample_data.reward_set[j])
        env.target_agent_cyber_env_reward_deque.append(np.mean(cyber_reward_data_this_step))

        env.sample_count += env.config.config_dict['SAMPLE_COUNT_PER_STEP']

        final_step_dynamics_train_loss = -1

        print("\nTrain for dynamics env----------------------")

        env.cyber_env.status = env.status_key['TRAIN']

        for i in range(env.config.config_dict['DYNAMICS_TRAIN_ITERATION']):
            data = env.real_env_sample_memory.sample(
                batch_size=env.cyber_env.model.config.config_dict['BATCH_SIZE'])

            final_step_dynamics_train_loss = env.cyber_env.fit(state_set=data['obs0'],
                                                               action_set=data['action'],
                                                               delta_state_label_set=data['delta'],
                                                               sess=tf.get_default_session())

        env.cyber_env.print_log_queue(env.status_key['TRAIN'])
        env.dyna_error_dequeu.append(final_step_dynamics_train_loss)
        if 'STE_V3_TEST_MOVE_OUT' in cfg.config_dict and cfg.config_dict['STE_V3_TEST_MOVE_OUT'] is True:
            pass
        else:
            progress_bar = np.floor(
                1.0 * env.target_agent._real_env_sample_count / env.config.config_dict['TEST_FRIQUENCY_SAMPLE'])
            if progress_bar > env.last_test:
                env.last_test = progress_bar

                # print("\nTest for dynamics env----------------------")
                # env.target_agent.status = env.status_key['TEST']
                # env.target_agent.env_status = env.target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS']
                # env.cyber_env.status = env.status_key['TEST']
                #
                # sample_data = env.target_agent.sample(env=env.test_env,
                #                                       sample_count=1000,
                #                                       store_flag=False,
                #                                       agent_print_log_flag=False)
                #
                # env.cyber_env.test(state_set=sample_data.state_set,
                #                    action_set=sample_data.action_set,
                #                    delta_state_label_set=np.array(sample_data.new_state_set) - np.array(
                #                        sample_data.state_set),
                #                    sess=tf.get_default_session())
                # env.cyber_env.print_log_queue(status=env.status_key['TEST'])

                print("\nTest for target agent by real cost function----------------------")
                env.target_agent.status = env.status_key['TEST']
                env.target_agent.env_status = env.target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS']

                assert env.target_agent.sampler.env_status == \
                       env.target_agent.sampler.config.config_dict['TEST_ENVIRONMENT_STATUS']

                sample_data = env.target_agent.sample(env=env.test_env,
                                                      sample_count=env.config.config_dict['TARGET_AGENT_TEST_SAMPLE'],
                                                      store_flag=False,
                                                      agent_print_log_flag=True)

        obs = 1.0
        reward = 0.0
        done = False
        info = [env.target_agent._real_env_sample_count, 0., train_in_cyber_flag]
        print("EEE env.target_agent._real_env_sample_count=", env.target_agent._real_env_sample_count)
        return obs, reward, done, info

    @staticmethod
    def trainer_env_step_v3_control_dyna_horizon(env, action):
        from src.env.trainerEnv.baselineTrainerEnv import BaselineTrainerEnv
        assert isinstance(env, BaselineTrainerEnv)
        env.step_count += 1

        F1 = action[0]
        prob_sample_on_real = action[2]
        prob_train_on_real = action[2]
        train_in_cyber_flag = False
        env.cyber_env.set_max_step(val=int(action[1] * env.cyber_env.max_step))

        print("\nTrain for target agent from real env----------------------")
        env.target_agent.status = env.status_key['TRAIN']

        t_r = int(max(env.config.config_dict['TARGET_AGENT_TRAIN_ITERATION'], 1))
        t_c = int((1 - prob_train_on_real) * t_r / prob_train_on_real)

        total_train = int(t_r + t_c)

        for i in range(total_train):
            prob = np.random.rand()
            if prob <= prob_train_on_real:
                env.target_agent.env_status = env.target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS']
                res_dict = env.target_agent.train()
            else:
                env.target_agent.env_status = env.target_agent.config.config_dict['CYBER_ENVIRONMENT_STATUS']
                res_dict = env.target_agent.train()
                if res_dict is not None:
                    train_in_cyber_flag = True

            env.target_agent.print_log_queue(status=env.status_key['TRAIN'])

        env.target_agent.sampler.set_F(F1=F1, F2=0.0)

        print("\n Real Env used count %d" % env.target_agent._real_env_sample_count)

        print("\nSample for target agent----------------------")

        env.target_agent.status = env.target_agent.status_key['TRAIN']
        env.target_agent.env_status = env.target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS']

        real_reward_data_this_step = []
        K_r = env.config.config_dict['SAMPLE_COUNT_PER_STEP']

        from src.util.sampler.fakeSampler import FakeSampler
        from src.util.sampler.fakeIntelligentSampler import FakeIntelligentSampler
        if isinstance(env.target_agent.sampler, (FakeSampler, FakeIntelligentSampler)):
            sample_count = K_r
            sample_step = 1
        else:
            sample_count = 1
            sample_step = K_r

        for i in range(sample_step):
            sample_data = env.target_agent.sample(env=env.real_env,
                                                  sample_count=sample_count,
                                                  store_flag=True,
                                                  agent_print_log_flag=True)

            for j in range(len(sample_data.state_set)):
                real_reward_data_this_step.append(sample_data.reward_set[j])

                data_dict = {
                    'obs0': sample_data.state_set[j],
                    'obs1': sample_data.new_state_set[j],
                    'action': sample_data.action_set[j],
                    'reward': sample_data.reward_set[j],
                    'terminal1': sample_data.done_set[j],
                    'delta_state': sample_data.new_state_set[j] - sample_data.state_set[j]
                }
                env.real_env_sample_memory.append(data_dict)
            env.cyber_env.model.update_mean_var(state_input=np.array(sample_data.state_set),
                                                action_input=np.array(sample_data.action_set),
                                                delta_state_label=np.array(sample_data.new_state_set) -
                                                                  np.array(sample_data.state_set))
        env.target_agent_real_env_reward_deque.append(np.mean(real_reward_data_this_step))

        cyber_reward_data_this_step = []
        K_c = int(
            env.config.config_dict['SAMPLE_COUNT_PER_STEP'] / prob_sample_on_real * (1.0 - prob_sample_on_real))
        if K_c < 1:
            K_c = 1

        if isinstance(env.target_agent.sampler, (FakeSampler, FakeIntelligentSampler)):
            sample_count = K_c
            sample_step = 1
        else:
            sample_count = 1
            sample_step = K_c

        env.target_agent.env_status = env.target_agent.config.config_dict['CYBER_ENVIRONMENT_STATUS']
        env.target_agent.status = env.target_agent.status_key['TRAIN']
        # env.target_agent.log_queue.queue.clear()
        for i in range(sample_step):
            sample_data = env.target_agent.sample(env=env.cyber_env,
                                                  sample_count=sample_count,
                                                  store_flag=True,
                                                  agent_print_log_flag=True)
            for j in range(len(sample_data.state_set)):
                cyber_reward_data_this_step.append(sample_data.reward_set[j])
        env.target_agent_cyber_env_reward_deque.append(np.mean(cyber_reward_data_this_step))

        env.sample_count += env.config.config_dict['SAMPLE_COUNT_PER_STEP']

        final_step_dynamics_train_loss = -1

        print("\nTrain for dynamics env----------------------")

        env.cyber_env.status = env.status_key['TRAIN']

        for i in range(env.config.config_dict['DYNAMICS_TRAIN_ITERATION']):
            data = env.real_env_sample_memory.sample(
                batch_size=env.cyber_env.model.config.config_dict['BATCH_SIZE'])

            final_step_dynamics_train_loss = env.cyber_env.fit(state_set=data['obs0'],
                                                               action_set=data['action'],
                                                               delta_state_label_set=data['delta'],
                                                               sess=tf.get_default_session())

            env.cyber_env.print_log_queue(env.status_key['TRAIN'])
        env.dyna_error_dequeu.append(final_step_dynamics_train_loss)

        progress_bar = np.floor(
            1.0 * env.target_agent._real_env_sample_count / env.config.config_dict['TEST_FRIQUENCY_SAMPLE'])
        if progress_bar > env.last_test:
            env.last_test = progress_bar
            print("\nTest for dynamics env----------------------")
            env.target_agent.status = env.status_key['TEST']
            env.target_agent.env_status = env.target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS']
            env.cyber_env.status = env.status_key['TEST']

            sample_data = env.target_agent.sample(env=env.test_env,
                                                  sample_count=1000,
                                                  store_flag=False,
                                                  agent_print_log_flag=False)

            env.cyber_env.test(state_set=sample_data.state_set,
                               action_set=sample_data.action_set,
                               delta_state_label_set=np.array(sample_data.new_state_set) - np.array(
                                   sample_data.state_set),
                               sess=tf.get_default_session())
            env.cyber_env.print_log_queue(status=env.status_key['TEST'])

            print("\nTest for target agent by real cost function----------------------")
            env.target_agent.status = env.status_key['TEST']
            env.target_agent.env_status = env.target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS']

            assert env.target_agent.sampler.env_status == \
                   env.target_agent.sampler.config.config_dict['TEST_ENVIRONMENT_STATUS']

            sample_data = env.target_agent.sample(env=env.test_env,
                                                  sample_count=env.config.config_dict['TARGET_AGENT_TEST_SAMPLE'],
                                                  store_flag=False,
                                                  agent_print_log_flag=True)

        obs = 0.0
        reward = 0.0
        done = False
        info = [env.target_agent._real_env_sample_count, 0., train_in_cyber_flag]
        print("EEE env.target_agent._real_env_sample_count=", env.target_agent._real_env_sample_count)
        return obs, reward, done, info

    def test(self, env):
        progress_bar = np.floor(
            1.0 * env.target_agent._real_env_sample_count / env.config.config_dict['TEST_FRIQUENCY_SAMPLE'])
        if progress_bar > env.last_test:
            env.last_test = progress_bar

            # print("\nTest for dynamics env----------------------")
            # env.target_agent.status = env.status_key['TEST']
            # env.target_agent.env_status = env.target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS']
            # env.cyber_env.status = env.status_key['TEST']
            #
            # sample_data = env.target_agent.sample(env=env.test_env,
            #                                       sample_count=1000,
            #                                       store_flag=False,
            #                                       agent_print_log_flag=False)
            #
            # env.cyber_env.test(state_set=sample_data.state_set,
            #                    action_set=sample_data.action_set,
            #                    delta_state_label_set=np.array(sample_data.new_state_set) - np.array(
            #                        sample_data.state_set),
            #                    sess=tf.get_default_session())
            # env.cyber_env.print_log_queue(status=env.status_key['TEST'])

            print("\nTest for target agent by real cost function----------------------")
            env.target_agent.status = env.status_key['TEST']
            env.target_agent.env_status = env.target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS']

            assert env.target_agent.sampler.env_status == \
                   env.target_agent.sampler.config.config_dict['TEST_ENVIRONMENT_STATUS']

            assert env.target_agent.sampler.env_status == env.target_agent.sampler.config.config_dict[
                'TEST_ENVIRONMENT_STATUS']

            sample_data = env.target_agent.sample(env=env.test_env,
                                                  sample_count=env.config.config_dict['TARGET_AGENT_TEST_SAMPLE'],
                                                  store_flag=False,
                                                  agent_print_log_flag=True)
