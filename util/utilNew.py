import os
import sys

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)
sys.path.append(CURRENT_PATH + '/../')
from src.env.util.utils import make_env
from util.classCreator.classCreator import *
import glob
from src.config.config import Config
import time
import json
from os.path import expanduser
import random


def load_all_config(config_path):
    res = []
    dir_list = glob.glob(pathname=config_path + '/**/*.json', recursive=True)
    for dir in dir_list:
        conf = Config.load_json(file_path=dir)
        res.append((dir, conf))
    return res


def create_tmp_config_file(game_env_name, orgin_config_path):
    all_config = load_all_config(config_path=orgin_config_path)
    tmp_path = expanduser('~') + '/.cache/tmpconfig/' + game_env_name + '/' + time.strftime("%Y-%m-%d_%H-%M-%S")
    while os.path.exists(tmp_path):
        tmp_path = tmp_path + '_' + str(random.randint(0, 100000))
    os.makedirs(tmp_path)
    for conf in all_config:
        file_name = os.path.basename(conf[0])
        with open(file=tmp_path + '/' + file_name, mode='w') as f:
            json.dump(conf[1], fp=f, sort_keys=True)
    return tmp_path


def create_baseline_game(env_id, game_specific_config_path, config_set_path, bound_file, target_model_type,
                         cost_fn=None,
                         done_fn=None,
                         cuda_device=0,
                         reset_fn=None,
                         sess=None,
                         name_prefix='',
                         exp_log_end=''):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    if sess is None:
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        sess.__enter__()

    real_env, test_env, bound, action_bound, obs_bound = _basic_init(env_id=env_id, bound_file=bound_file)

    cyber_env = create_dynamics_env(config_path=config_set_path + '/dynamicsEnvTestConfig.json',
                                    sess=tf.get_default_session(),
                                    dyna_model=create_dynamics_env_mlp_model(
                                        config_path=game_specific_config_path + '/dynamicsEnvMlpModelTestConfig.json',
                                        update_dict={**_generate_state_action_bound(real_env),
                                                     **{"NAME": name_prefix + "dynamics_model"}},
                                        output_bound=obs_bound),
                                    real_env=real_env,
                                    cost_fn=cost_fn,
                                    done_fn=done_fn,
                                    reset_fn=reset_fn,
                                    update_dict={**_generate_state_action_bound(real_env), **{"NAME": "dynamics_env"}})

    target_agent_model = _create_different_type_target_agent_model(target_model_type=target_model_type,
                                                                   game_specific_config_path=game_specific_config_path,
                                                                   action_bound=action_bound,
                                                                   obs_bound=obs_bound,
                                                                   update_dict={
                                                                       **_generate_state_action_bound(real_env),
                                                                       **{"NAME": name_prefix + "target_agent_model"}},
                                                                   real_env=real_env)

    target_agent = create_target_agent(config_path=config_set_path + '/ddpgAgentTestConfig.json',
                                       sampler=Sampler(cost_fn=cost_fn),
                                       real_env=real_env,
                                       dyna_env=cyber_env,
                                       model=target_agent_model,
                                       update_dict={"NAME": name_prefix + "target agent"})

    baseline_trainer_env = create_baseline_trainer_env(
        config_path=config_set_path + '/baselineTrainerEnvTestConfig.json',
        dyna_env=cyber_env,
        real_env=real_env,
        target_agent=target_agent,
        test_env=test_env,
        update_dict={"NAME": name_prefix + "target baselineline trainer env"})
    baseline_trainer_agent = create_baseline_trainer_agent(
        config_path=config_set_path + '/baselineTrainerAgentTestConfig.json',
        update_dict={"NAME": name_prefix + "target baselineline trainer agent"},
        model=create_fix_output_model(config_path=config_set_path + '/baselineTrainerModelTestConfig.json',
                                      update_dict=None),
        env=baseline_trainer_env)
    player = create_game_player(config_path=config_set_path + '/gamePlayerTestConfig.json',
                                update_dict={'GAME_NAME': env_id, "NAME": name_prefix + "player"},
                                basic_list=[cyber_env.model, cyber_env, target_agent.model, target_agent,
                                            baseline_trainer_env, baseline_trainer_agent.model, baseline_trainer_agent],
                                env=baseline_trainer_env,
                                agent=baseline_trainer_agent,
                                experiment_type=0,
                                log_path_end=exp_log_end)
    return player, sess


def create_intelligent_game(env_id, game_specific_config_path, bound_file, target_model_type, config_set_path,
                            intelligent_model_type,
                            cost_fn=None, cuda_device=0,
                            done_fn=None,
                            reset_fn=None,
                            sess=None,
                            refer_intel_player=None,
                            name_prefix='',
                            exp_end_with=''):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    if sess is None:
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        sess.__enter__()

    real_env, test_env, bound, action_bound, obs_bound = _basic_init(env_id=env_id, bound_file=bound_file)

    cyber_env = create_dynamics_env(config_path=config_set_path + '/dynamicsEnvTestConfig.json',
                                    sess=tf.get_default_session(),
                                    dyna_model=create_dynamics_env_mlp_model(
                                        config_path=game_specific_config_path + '/dynamicsEnvMlpModelTestConfig.json',
                                        update_dict={**_generate_state_action_bound(real_env),
                                                     **{"NAME": name_prefix + "dynamics_model"}},
                                        output_bound=obs_bound),
                                    real_env=real_env,
                                    cost_fn=cost_fn,
                                    done_fn=done_fn,
                                    reset_fn=reset_fn,
                                    update_dict={**_generate_state_action_bound(real_env), **{"NAME": "dynamics_env"}})

    target_agent_model = _create_different_type_target_agent_model(target_model_type=target_model_type,
                                                                   game_specific_config_path=game_specific_config_path,
                                                                   action_bound=action_bound,
                                                                   obs_bound=obs_bound,
                                                                   update_dict={
                                                                       **_generate_state_action_bound(real_env),
                                                                       **{"NAME": name_prefix + "target_agent_model"}},
                                                                   real_env=real_env)
    if refer_intel_player is not None:
        if intelligent_model_type == 'RANDOM':
            sampler = create_fake_intelligent_sampler(config_path=config_set_path + '/intelligentSamplerConfig.json',
                                                      cost_fn=cost_fn,
                                                      update_dict={"NAME": name_prefix + "fake_intelligent_sampler"},
                                                      env=refer_intel_player.env)
        else:
            sampler = create_fake_sampler(cost_fn=cost_fn, env=refer_intel_player.env, config_path=None,
                                          update_dict=None)
    else:
        sampler = create_intelligent_sampler(config_path=config_set_path + '/intelligentSamplerConfig.json',
                                             cost_fn=cost_fn, update_dict={"NAME": name_prefix + "intelligent_sampler"})
    target_agent = create_target_agent(config_path=config_set_path + '/ddpgAgentTestConfig.json',
                                       update_dict={"NAME": name_prefix + "target_agent"},
                                       sampler=sampler,
                                       real_env=real_env,
                                       dyna_env=cyber_env,
                                       model=target_agent_model)

    intelligent_trainer_env = create_trainer_env(config_path=config_set_path + '/trainerEnvTestConfig.json',
                                                 dyna_env=cyber_env,
                                                 real_env=real_env,
                                                 test_env=test_env,
                                                 target_agent=target_agent,
                                                 update_dict={"NAME": name_prefix + "target_intelligent_trainer_env"})

    intelligent_trainer_agent = _create_different_type_intelligent_trainer_agent(
        intelligent_trainer_agent_type=intelligent_model_type,
        config_set_path=config_set_path,
        intelligent_trainer_env=intelligent_trainer_env,
        action_bound=(intelligent_trainer_env.action_space.low, intelligent_trainer_env.action_space.high),
        update_dict={"NAME": name_prefix + "target_intelligent_trainer_agent"})

    player = create_game_player(config_path=config_set_path + '/gamePlayerTestConfig.json',
                                update_dict={'GAME_NAME': env_id, "NAME": name_prefix + "player"},
                                basic_list=[cyber_env.model, cyber_env, target_agent.model, target_agent,
                                            intelligent_trainer_env, intelligent_trainer_agent.model,
                                            intelligent_trainer_agent],
                                env=intelligent_trainer_env,
                                agent=intelligent_trainer_agent,
                                experiment_type=1,
                                refer_player=refer_intel_player,
                                name=name_prefix,
                                log_path_end=exp_end_with)
    return player, sess


def create_random_ensemble_game(env_id,
                                bound_file,
                                player_config_path_list,
                                player_target_model_type_list,
                                cost_fn=None,
                                cuda_device=0,
                                intelligent_agent_index=None,
                                done_fn=None,
                                reset_fn=None,
                                exp_end_with=""):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.__enter__()
    player_list = []
    if intelligent_agent_index is not None:
        intel_player, _ = create_intelligent_game(env_id=env_id,
                                                  config_set_path=player_config_path_list[intelligent_agent_index][0],
                                                  game_specific_config_path=
                                                  player_config_path_list[intelligent_agent_index][1],
                                                  bound_file=bound_file,
                                                  intelligent_model_type=
                                                  player_target_model_type_list[intelligent_agent_index][0],
                                                  target_model_type=
                                                  player_target_model_type_list[intelligent_agent_index][1],
                                                  cost_fn=cost_fn,
                                                  cuda_device=cuda_device,
                                                  done_fn=done_fn,
                                                  sess=sess,
                                                  reset_fn=reset_fn,
                                                  name_prefix='Intel1',
                                                  exp_end_with=exp_end_with)
        player_list.append(intel_player)

        player, _ = create_baseline_with_intel_trainer_env_game(env_id=env_id,
                                                                config_set_path=
                                                                player_config_path_list[1][0],
                                                                game_specific_config_path=
                                                                player_config_path_list[1][1],
                                                                bound_file=bound_file,
                                                                target_model_type=
                                                                player_target_model_type_list[1][1],
                                                                cost_fn=cost_fn,
                                                                cuda_device=cuda_device,
                                                                done_fn=done_fn,
                                                                reset_fn=reset_fn,
                                                                sess=sess,
                                                                name_prefix=player_target_model_type_list[1][
                                                                                0] + '_' + str(1),
                                                                exp_log_end=exp_end_with)
        player.logger._log_dir = intel_player.logger.log_dir + '_' + player_target_model_type_list[1][
            0] + '_' + str(1)
        player_list.append(player)

        intel_player2, _ = create_intelligent_game(env_id=env_id,
                                                   config_set_path=player_config_path_list[2][0],
                                                   game_specific_config_path=
                                                   player_config_path_list[2][1],
                                                   bound_file=bound_file,
                                                   intelligent_model_type=
                                                   player_target_model_type_list[2][0],
                                                   target_model_type=
                                                   player_target_model_type_list[2][1],
                                                   cost_fn=cost_fn,
                                                   cuda_device=cuda_device,
                                                   done_fn=done_fn,
                                                   sess=sess,
                                                   reset_fn=reset_fn,
                                                   name_prefix='Intel2',
                                                   exp_end_with=exp_end_with)
        intel_player2.logger._log_dir = intel_player.logger.log_dir + '_' + player_target_model_type_list[2][
            0] + '_' + str(2)
        player_list.append(intel_player2)

        # create fake sampler for memory sharing
        fakeSamplers = []
        for i in range(len(player_list)):
            sampler = create_fake_intelligent_sampler(
                config_path=player_config_path_list[i][0] + '/intelligentSamplerConfig.json',
                cost_fn=cost_fn,
                update_dict={"NAME": player_target_model_type_list[i][0] + '_' + str(i) + "fake_intelligent_sampler"},
                env=player_list[i].env)
            fakeSamplers.append(sampler)

        assemble = create_random_ensemble_player(player_list=player_list, intel_index=intelligent_agent_index,
                                                 fakeSamplers=fakeSamplers)
        return assemble, sess


def _basic_init(env_id, bound_file):
    real_env = make_env(env_id)
    test_env = make_env(env_id)
    import config as cfg

    if 'SWIMMER_HORIZON' in cfg.config_dict and env_id == 'Swimmer-v1':
        real_env._max_episode_steps = cfg.config_dict['SWIMMER_HORIZON']
    bound = Config.load_json(file_path=bound_file)

    action_bound = (np.array(bound['ACTION_LOW']), np.array(bound['ACTION_HIGH']))
    obs_bound = (np.array(bound['STATE_LOW']), np.array(bound['STATE_HIGH']))

    return real_env, test_env, bound, action_bound, obs_bound


def _generate_state_action_bound(real_env):
    space_dict = {
        "STATE_SPACE": tuple(real_env.observation_space.shape),
        "ACTION_SPACE": tuple(real_env.action_space.shape),
    }
    return space_dict


def _create_different_type_target_agent_model(target_model_type, game_specific_config_path, action_bound, obs_bound,
                                              real_env, update_dict=None):
    if update_dict is None:
        update_dict = _generate_state_action_bound(real_env)
    if target_model_type == 'TRPO':
        target_agent_model = create_trpo(config_path=game_specific_config_path + '/targetModelTestConfig.json',
                                         action_bound=action_bound,
                                         obs_bound=obs_bound,
                                         update_dict=update_dict)
    elif target_model_type == 'DDPG':
        target_agent_model = create_ddpg(config_path=game_specific_config_path + '/targetModelTestConfig.json',
                                         action_bound=action_bound,
                                         obs_bound=obs_bound,
                                         update_dict=update_dict)
    else:
        raise KeyError('Wrong type of target agent model')
    return target_agent_model


def _create_different_type_intelligent_trainer_model(intelligent_trainer_agent_type, config_set_path, action_bound,
                                                     update_dict=None):
    if intelligent_trainer_agent_type == 'DQN':
        model = create_dqn(config_path=config_set_path,
                           action_bound=action_bound,
                           update_dict=update_dict)
    elif intelligent_trainer_agent_type == 'REINFORCE':
        model = create_reinforce_model(config_path=config_set_path,
                                       action_bound=action_bound,
                                       update_dict=update_dict)
    elif intelligent_trainer_agent_type == 'FIX':
        model = create_fix_output_model(config_path=config_set_path,
                                        update_dict=update_dict)

    else:
        raise KeyError('Wrong type of intelligent agent model')

    return model


def _create_different_type_intelligent_trainer_agent(intelligent_trainer_agent_type, intelligent_trainer_env,
                                                     config_set_path, action_bound,
                                                     update_dict=None):
    if intelligent_trainer_agent_type != 'RANDOM':
        model = _create_different_type_intelligent_trainer_model(
            intelligent_trainer_agent_type=intelligent_trainer_agent_type,
            config_set_path=config_set_path + '/intelligentTrainerModelTestConfig.json',
            action_bound=action_bound,
            update_dict=update_dict)

        intelligent_trainer_agent = create_intelligent_trainer_agent(
            config_path=config_set_path + '/intelligentTrainerAgentTestConfig.json',
            model=model,
            update_dict=update_dict,
            env=intelligent_trainer_env)
    else:
        intelligent_trainer_agent = create_intelligent_random_trainer_agent(
            config_path=config_set_path + '/intelligentTrainerAgentTestConfig.json',
            update_dict=update_dict,
            env=intelligent_trainer_env)
    return intelligent_trainer_agent


def create_baseline_with_intel_trainer_env_game(env_id, game_specific_config_path, config_set_path, bound_file,
                                                target_model_type,
                                                cost_fn=None,
                                                done_fn=None,
                                                cuda_device=0,
                                                reset_fn=None,
                                                sess=None,
                                                name_prefix='',
                                                exp_log_end=''):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    if sess is None:
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        sess.__enter__()

    real_env, test_env, bound, action_bound, obs_bound = _basic_init(env_id=env_id, bound_file=bound_file)

    cyber_env = create_dynamics_env(config_path=config_set_path + '/dynamicsEnvTestConfig.json',
                                    sess=tf.get_default_session(),
                                    dyna_model=create_dynamics_env_mlp_model(
                                        config_path=game_specific_config_path + '/dynamicsEnvMlpModelTestConfig.json',
                                        update_dict={**_generate_state_action_bound(real_env),
                                                     **{"NAME": name_prefix + "dynamics_model"}},
                                        output_bound=obs_bound),
                                    real_env=real_env,
                                    cost_fn=cost_fn,
                                    done_fn=done_fn,
                                    reset_fn=reset_fn,
                                    update_dict={**_generate_state_action_bound(real_env), **{"NAME": "dynamics_env"}})

    target_agent_model = _create_different_type_target_agent_model(target_model_type=target_model_type,
                                                                   game_specific_config_path=game_specific_config_path,
                                                                   action_bound=action_bound,
                                                                   obs_bound=obs_bound,
                                                                   update_dict={
                                                                       **_generate_state_action_bound(real_env),
                                                                       **{"NAME": name_prefix + "target_agent_model"}},
                                                                   real_env=real_env)

    target_agent = create_target_agent(config_path=config_set_path + '/ddpgAgentTestConfig.json',
                                       sampler=Sampler(cost_fn=cost_fn),
                                       real_env=real_env,
                                       dyna_env=cyber_env,
                                       model=target_agent_model,
                                       update_dict={"NAME": name_prefix + "target agent"})

    intelligent_trainer_env = create_trainer_env(config_path=config_set_path + '/trainerEnvTestConfig.json',
                                                 dyna_env=cyber_env,
                                                 real_env=real_env,
                                                 test_env=test_env,
                                                 target_agent=target_agent,
                                                 update_dict={"NAME": name_prefix + "target_intelligent_trainer_env"})
    baseline_trainer_agent = create_baseline_trainer_agent(
        config_path=config_set_path + '/baselineTrainerAgentTestConfig.json',
        update_dict={"NAME": name_prefix + "target baselineline trainer agent"},
        model=create_fix_output_model(config_path=config_set_path + '/baselineTrainerModelTestConfig.json',
                                      update_dict=None),
        env=intelligent_trainer_env)
    player = create_game_player(config_path=config_set_path + '/gamePlayerTestConfig.json',
                                update_dict={'GAME_NAME': env_id, "NAME": name_prefix + "player"},
                                basic_list=[cyber_env.model, cyber_env, target_agent.model, target_agent,
                                            intelligent_trainer_env, baseline_trainer_agent.model,
                                            baseline_trainer_agent],
                                env=intelligent_trainer_env,
                                agent=baseline_trainer_agent,
                                experiment_type=0,
                                log_path_end=exp_log_end)
    return player, sess


if __name__ == '__main__':
    from src.env.util import *
    from conf.envBound import get_bound_file

    game_env_name = 'Pendulum-v0'
    config_set_path = "/home/dls/CAP/intelligenttrainerframework/conf/configSet_Pendulum/"
    game_specific_config_path = '/home/dls/CAP/intelligenttrainerframework/conf/configSet_Pendulum/modelNetworkConfig/invertedPendulum/'

    player, sess = create_baseline_game(env_id=game_env_name,
                                        game_specific_config_path=game_specific_config_path,
                                        config_set_path=config_set_path,
                                        bound_file=get_bound_file(env_name=game_env_name),
                                        target_model_type='DDPG',
                                        cost_fn=COST_FUNCTION_ENV_DICT[game_env_name],
                                        done_fn=DONE_FUNCTION_ENV_DICT[game_env_name])
    player.play()
