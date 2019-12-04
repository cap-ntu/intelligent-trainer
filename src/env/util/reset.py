import numpy as np
from src.env.util.utils import half_cheetah_instance, reacher_instance, inverted_pendulum_instance, swimmer_instance#, \
    # CFD_instance


def half_cheetah_reset_func():
    # nq = 9
    # nv = 9
    #
    # qpos = half_cheetah_instance.init_qpos + np.random.uniform(low=-.1, high=.1, size=nq)
    # qvel = half_cheetah_instance.init_qvel + np.random.randn(nv) * .1
    # return np.concatenate((qpos.flat[1:], qvel.flat))
    return half_cheetah_instance.reset()


def continuous_mountain_car_reset_function():
    return np.array([np.random.uniform(low=-0.6, high=-0.4), 0])


def reacher_reset_function():
    # qpos = np.random.uniform(low=-0.1, high=0.1, size=reacher_instance.unwrapped.model.nq)
    # init_qpos = reacher_instance.init_qpos
    # init_qvel = reacher_instance.init_qvel
    # while True:
    #     goal = np.random.uniform(low=-.2, high=.2, size=2)
    #     if np.linalg.norm(goal) < 2:
    #         break
    # qpos[-2:] = goal
    # qvel = init_qvel + np.random.uniform(low=-.005, high=.005, size=reacher_instance.unwrapped.model.nv)
    # qvel[-2:] = 0
    # theta = qpos.flat[:2]
    #
    # return np.concatenate((np.cos(theta),
    #                        np.sin(theta),
    #                        qpos.flat[2:],
    #                        qvel.flat[:2]
    #                        qpos, qvel))
    return reacher_instance.reset()


def inverted_pendulum_reset_function():
    return np.concatenate((inverted_pendulum_instance.unwrapped.init_qpos + np.random.uniform(low=-.1, high=.1, size=2),
                           inverted_pendulum_instance.unwrapped.init_qvel + np.random.randn(2) * .1))


def pendulum_reset_function():
    high = np.array([np.pi, 1])
    state = np.random.uniform(low=-high, high=high)
    theta = state[0]
    thetadot = state[1]
    return np.array([np.cos(theta), np.sin(theta), thetadot])


def swimmer_reset_function():
    return swimmer_instance.reset()


# def CFD_reset_function():
#     return CFD_instance.reset()


RESET_FUNCTION_ENV_DICT = {
    "MountainCarContinuous-v0": continuous_mountain_car_reset_function,
    "Pendulum-v0": pendulum_reset_function,
    "Ant-v1": None,
    "HumanoidStandup-v1": None,
    "Swimmer-v1": swimmer_reset_function,
    "HalfCheetah": half_cheetah_reset_func,
    "Reacher-v1": reacher_reset_function,
    # "CFD": CFD_reset_function,
    "InvertedPendulum-v1": inverted_pendulum_reset_function
}
