import os
import sys
import tensorflow as tf
import numpy as np

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)
sys.path.append(CURRENT_PATH + '/../')
from src.player import GamePlayer
from src.config.config import Config
from src.model.ddpgModel.ddpgModel import DDPGModel
from src.model.dqnModel.dqnModel import DQNModel
from src.model.trpoModel.trpoModel import TrpoModel
from src.model.fxiedOutputModel.FixedOutputModel import FixedOutputModel
from src.model.dynamicsEnvMlpModel.dynamicsEnvMlpModel import DynamicsEnvMlpModel
from src.model.reinforceModel.reinforceModel import REINFORCEModel
from src.env.trainerEnv.baselineTrainerEnv import BaselineTrainerEnv
from src.env.dynamicsEnv.dynamicsEnv import DynamicsEnv
from src.env.trainerEnv.trainerEnv import TrainerEnv
from src.agent.targetAgent.targetAgent import TargetAgent
from src.agent.baselineTrainerAgent.baselineTrainerAgent import BaselineTrainerAgent
from src.agent.IntelligentTrainerAgent.intelligentRandomTrainerAgent import IntelligentRandomTrainerAgent
from src.agent.IntelligentTrainerAgent.intelligentTrainerAgent import IntelligentTrainerAgent
from src.util.sampler.sampler import Sampler
from src.util.sampler.intelligentSampler import IntelligentSampler
from src.util.sampler.fakeSampler import FakeSampler
from src.util.sampler.fakeIntelligentSampler import FakeIntelligentSampler
from src.player import RandomEnsemblePlayer
from src.model.model import Model


def create_trpo(config_path, action_bound, obs_bound, update_dict):
    trpo_config = load_config(key_list=TrpoModel.key_list, config_path=config_path, update_dict=update_dict)
    trpo = TrpoModel(config=trpo_config, action_bound=action_bound, obs_bound=obs_bound)
    return trpo


def create_ddpg(config_path, action_bound, obs_bound, update_dict):
    ddpg_conf = load_config(key_list=DDPGModel.key_list, config_path=config_path, update_dict=update_dict)
    ddpg = DDPGModel(config=ddpg_conf, action_bound=action_bound, obs_bound=obs_bound)
    return ddpg


def create_dqn(config_path, action_bound, update_dict):
    dqn_conf = load_config(key_list=DQNModel.key_list, config_path=config_path, update_dict=update_dict)

    dqn_model = DQNModel(config=dqn_conf,
                         action_bound=action_bound)
    return dqn_model


def create_dynamics_env_mlp_model(config_path, output_bound, update_dict):
    dynamics_model_conf = load_config(key_list=DynamicsEnvMlpModel.key_list,
                                      config_path=config_path,
                                      update_dict=update_dict)

    dynamics_model = DynamicsEnvMlpModel(config=dynamics_model_conf, output_bound=output_bound)
    return dynamics_model


def create_fix_output_model(config_path, update_dict):
    fix_model_conf = load_config(key_list=FixedOutputModel.key_list, config_path=config_path, update_dict=update_dict)
    fix_output_model = FixedOutputModel(config=fix_model_conf)
    return fix_output_model


def create_reinforce_model(config_path, action_bound, update_dict):
    reinforce_model_conf = load_config(key_list=REINFORCEModel.key_list,
                                       config_path=config_path,
                                       update_dict=update_dict)
    reinforce_model = REINFORCEModel(config=reinforce_model_conf,
                                     action_bound=action_bound)
    return reinforce_model


def create_dynamics_env(config_path, dyna_model, sess, real_env, cost_fn, done_fn, reset_fn, update_dict):
    dynamics_env_conf = load_config(key_list=DynamicsEnv.key_list,
                                    config_path=config_path,
                                    update_dict=update_dict)

    dyna_env = DynamicsEnv(config=dynamics_env_conf, sess=sess, model=dyna_model, init_env=real_env,
                           cost=cost_fn, done=done_fn, reset=reset_fn)
    return dyna_env


def create_trainer_env(config_path, dyna_env, real_env, target_agent, test_env, update_dict):
    trainer_env_conf = load_config(key_list=TrainerEnv.key_list,
                                   config_path=config_path,
                                   update_dict=update_dict)
    env = TrainerEnv(config=trainer_env_conf,
                     cyber_env=dyna_env,
                     real_env=real_env,
                     target_agent=target_agent,
                     test_env=test_env)
    return env


def create_baseline_trainer_env(config_path, dyna_env, real_env, target_agent, test_env, update_dict):
    trainer_env_conf = load_config(key_list=BaselineTrainerEnv.key_list,
                                   config_path=config_path,
                                   update_dict=update_dict)
    env = BaselineTrainerEnv(config=trainer_env_conf,
                             cyber_env=dyna_env,
                             real_env=real_env,
                             target_agent=target_agent,
                             test_env=test_env)
    return env


def create_target_agent(config_path, update_dict, sampler, real_env, dyna_env, model):
    target_agent_config = load_config(key_list=TargetAgent.key_list,
                                      config_path=config_path,
                                      update_dict=update_dict)

    a = TargetAgent(config=target_agent_config,
                    real_env=real_env,
                    cyber_env=dyna_env,
                    model=model,
                    sampler=sampler)
    return a


def create_baseline_trainer_agent(config_path, update_dict, model, env):
    baseline_trainer_agent_config = load_config(key_list=BaselineTrainerAgent.key_list,
                                                config_path=config_path,
                                                update_dict=update_dict)

    trainer_agent = BaselineTrainerAgent(config=baseline_trainer_agent_config,
                                         model=model,
                                         env=env)
    return trainer_agent


def create_intelligent_trainer_agent(config_path, update_dict, model, env):
    intelligent_trainer_agent_config = load_config(key_list=IntelligentTrainerAgent.key_list,
                                                   config_path=config_path,
                                                   update_dict=update_dict)

    trainer_agent = IntelligentTrainerAgent(config=intelligent_trainer_agent_config,
                                            model=model,
                                            env=env)
    return trainer_agent


def create_intelligent_random_trainer_agent(config_path, update_dict, env):
    random_trainer_config = load_config(key_list=IntelligentRandomTrainerAgent.key_list,
                                        config_path=config_path,
                                        update_dict=update_dict)
    trainer_agent = IntelligentRandomTrainerAgent(config=random_trainer_config,
                                                  model=Model(config=None),
                                                  env=env)
    return trainer_agent


def create_intelligent_sampler(config_path, update_dict, cost_fn):
    sampler_conf = load_config(key_list=IntelligentSampler.key_list,
                               config_path=config_path,
                               update_dict=update_dict)

    sampler = IntelligentSampler(cost_fn=cost_fn, config=sampler_conf)
    return sampler


def create_sampler(config_path, update_dict):
    return Sampler()


def create_fake_sampler(config_path, update_dict, cost_fn, env):
    sampler = FakeSampler(cost_fn=cost_fn, config=None, reference_trainer_env=env)
    return sampler


def create_fake_intelligent_sampler(config_path, update_dict, cost_fn, env):
    sample_conf = load_config(key_list=IntelligentSampler.key_list,
                              config_path=config_path,
                              update_dict=update_dict)
    sampler = FakeIntelligentSampler(config=sample_conf, cost_fn=cost_fn, reference_trainer_env=env)
    return sampler


def create_game_player(config_path, update_dict, env, agent, experiment_type, basic_list, refer_player=None, name='',
                       log_path_end=''):
    player_config = load_config(key_list=GamePlayer.key_list,
                                config_path=config_path,
                                update_dict=update_dict)
    if refer_player:

        player = GamePlayer(config=player_config, env=env, agent=agent, basic_list=basic_list,
                            ep_type=experiment_type, log_path=refer_player.logger.log_dir + name,
                            log_path_end_with=log_path_end)
    else:
        player = GamePlayer(config=player_config, env=env, agent=agent, basic_list=basic_list,
                            ep_type=experiment_type, log_path_end_with=log_path_end)
    return player

#
# def create_assemble_player(main_player, ref_player_list):
#     assemble_player = AssembleGamePlayer(intel_player=main_player, ref_player_list=ref_player_list)
#     return assemble_player


def create_random_ensemble_player(player_list, intel_index, fakeSamplers):
    p = RandomEnsemblePlayer(player_list=player_list, intel_trainer_index=intel_index, fakeSamplers = fakeSamplers)
    return p


def update_config_dict(config, dict):
    if dict is None:
        return
    for key, value in dict.items():
        config.config_dict[key] = value


def load_config(key_list, config_path, update_dict):
    conf = Config(standard_key_list=key_list)
    conf.load_config(path=config_path)
    update_config_dict(config=conf, dict=update_dict)
    return conf
