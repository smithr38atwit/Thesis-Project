import os
from enum import Enum


class Algs(Enum):
    SEAC = 0
    MAPPO = 1


ALGORITHM = Algs.MAPPO
EVAL = False

match ALGORITHM:
    case Algs.SEAC:
        os.system(
            "python seac/train.py with env_name=rware-small-4ag-v1 num_env_steps=20000000.0 time_limit=500 transfer=True eval_interval=None"
        )
    case Algs.MAPPO:
        if EVAL:
            os.system(
                'python epymarl-main/src/main.py --config=mappo --env-config=gymma with env_args.key="rware:rware-tiny-2ag-v1" evaluate=True'
            )
        else:
            os.system(
                'python epymarl-main/src/main.py --config=mappo --env-config=gymma with env_args.key="rware:rware-tiny-2ag-v1"'
            )
