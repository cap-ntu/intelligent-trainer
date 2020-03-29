from log.baselineTestLog import LOG
from log.intelligentTestLog import INTEL_LOG
import json
import os
import numpy as np


def compare_run_time(algo_path_dict: dict, normalize_by_algo):
    res = {}
    for algo_name, base_path_list in algo_path_dict.items():
        res[algo_name] = []
        for p in base_path_list:
            config_dict = json.load(os.path.join(p, 'conf', 'GamePlayer.json'))
            res[algo_name].append(config_dict['START_TIME'] - config_dict['END_TIME'])
        res[algo_name] = int(np.mean(res[algo_name]))
    for algo_name, val in res.items():
        res[algo_name] /= res[normalize_by_algo]
    json.dumps(res, indent=4)
    return res
