#!/usr/bin/env bash

# no dyna
source activate py3.5 && cd /home/dls/CAP/intelligenttrainerpublic && python test/testBaselineNoDyna.py --env Pendulum-v0 --cuda_id 0 --num 5
source activate py3.5 && cd /home/dls/CAP/intelligenttrainerpublic && python test/testBaselineNoDyna.py --env MountainCarContinuous-v0 --cuda_id 0 --num 5
source activate py3.5 && cd /home/dls/CAP/intelligenttrainerpublic && python test/testBaselineNoDyna.py --env Reacher-v1 --cuda_id 0 --num 5
source activate py3.5 && cd /home/dls/CAP/intelligenttrainerpublic && python test/testBaselineNoDyna.py --env HalfCheetah --cuda_id 0 --num 5
source activate py3.5 && cd /home/dls/CAP/intelligenttrainerpublic && python test/testBaselineNoDyna.py --env Swimmer-v1 --cuda_id 0 --num 5

# dqn trainer
source activate py3.5 && cd /home/dls/CAP/intelligenttrainerpublic && python test/testBaselineNoDyna.py --env Pendulum-v0 --cuda_id 0 --num 5
source activate py3.5 && cd /home/dls/CAP/intelligenttrainerpublic && python test/testBaselineNoDyna.py --env MountainCarContinuous-v0 --cuda_id 0 --num 5
source activate py3.5 && cd /home/dls/CAP/intelligenttrainerpublic && python test/testBaselineNoDyna.py --env Reacher-v1 --cuda_id 0 --num 5
source activate py3.5 && cd /home/dls/CAP/intelligenttrainerpublic && python test/testBaselineNoDyna.py --env HalfCheetah --cuda_id 0 --num 5
source activate py3.5 && cd /home/dls/CAP/intelligenttrainerpublic && python test/testBaselineNoDyna.py --env Swimmer-v1 --cuda_id 0 --num 5

