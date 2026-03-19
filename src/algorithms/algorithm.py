from omegaconf import open_dict

from src.algorithms import (
    ac_lambda,
    dqn,
    gradient_ppo,
    mappo,
    ppo,
    pqn,
    r2d2,
    sac,
)

register = {
    "ppo": ppo.make,
    "pqn": pqn.make,
    "dqn": dqn.make,
    "sac": sac.make,
    "ac_lambda": ac_lambda.make,
    "gradient_ppo": gradient_ppo.make,
    "mappo": mappo.make,
    "r2d2": r2d2.make,
}


def make(cfg, env, env_params):
    name = cfg.algorithm.name
    with open_dict(cfg):
        del cfg.algorithm.name
        del cfg.stack.name
    return register[name](cfg, env, env_params)
