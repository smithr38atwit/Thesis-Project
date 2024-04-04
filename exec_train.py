import os
from enum import Enum


class Algs(Enum):
    SEAC = 0
    Q_LEARN = 1


ALGORITHM = Algs.Q_LEARN
EVAL = False

match ALGORITHM:
    case Algs.SEAC:
        os.system(
            "python seac/train.py with env_name=rware-tiny-2ag-v1 num_env_steps=20000000.0 time_limit=500 transfer=False"
        )
    case Algs.Q_LEARN:
        if EVAL:
            os.system(
                'python epymarl-main/src/main.py --config=qmix --env-config=gymma with env_args.key="rware:rware-tiny-2ag-v1" evaluate=True'
            )
        else:
            os.system(
                'python epymarl-main/src/main.py --config=qmix --env-config=gymma with env_args.key="rware:rware-tiny-2ag-v1"'
            )
