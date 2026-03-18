from omegaconf import open_dict

from src.algorithms import (
    ppo_brax,
    ppo_pobax,
    pqn_craftax,
    pqn_grimax,
    pqn_minatar,
    pqn_popgym_arcade,
    pqn_popjym,
)

register = {
    ("ppo", "brax"): ppo_brax.make,
    ("ppo", "pobax"): ppo_pobax.make,
    ("pqn", "grimax"): pqn_grimax.make,
    ("pqn", "minatar"): pqn_minatar.make,
    ("pqn", "popjym"): pqn_popjym.make,
    ("pqn", "popgym_arcade"): pqn_popgym_arcade.make,
    ("pqn", "craftax"): pqn_craftax.make,
    ("pqn", "navix"): pqn_grimax.make,
}


def make(cfg, env, env_params):
    family = cfg.environment.get("family", cfg.environment.namespace)
    name = cfg.algorithm.name
    with open_dict(cfg):
        del cfg.algorithm.name
    key = (name, family)
    return register[key](cfg, env, env_params)
