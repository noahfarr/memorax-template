from omegaconf import open_dict

from src.algorithms import (
    ac_lambda_brax,
    ac_lambda_popjym,
    dqn_minatar,
    dqn_popgym_arcade,
    dqn_popjym,
    ppo_brax,
    ppo_mujoco_playground,
    ppo_pobax,
    pqn_craftax,
    pqn_grimax,
    pqn_minatar,
    pqn_popgym_arcade,
    pqn_popjym,
    pqn_xminigrid,
    sac_brax,
    sac_mujoco_playground,
)

register = {
    ("ppo", "brax"): ppo_brax.make,
    ("ppo", "pobax"): ppo_pobax.make,
    ("ppo", "mujoco_playground"): ppo_mujoco_playground.make,
    ("pqn", "grimax"): pqn_grimax.make,
    ("pqn", "minatar"): pqn_minatar.make,
    ("pqn", "popjym"): pqn_popjym.make,
    ("pqn", "popgym_arcade"): pqn_popgym_arcade.make,
    ("pqn", "craftax"): pqn_craftax.make,
    ("pqn", "navix"): pqn_grimax.make,
    ("pqn", "xminigrid"): pqn_xminigrid.make,
    ("dqn", "minatar"): dqn_minatar.make,
    ("dqn", "popjym"): dqn_popjym.make,
    ("dqn", "popgym_arcade"): dqn_popgym_arcade.make,
    ("sac", "brax"): sac_brax.make,
    ("sac", "mujoco_playground"): sac_mujoco_playground.make,
    ("ac_lambda", "popjym"): ac_lambda_popjym.make,
    ("ac_lambda", "brax"): ac_lambda_brax.make,
}


def make(cfg, env, env_params):
    family = cfg.environment.get("family", cfg.environment.namespace)
    name = cfg.algorithm.name
    with open_dict(cfg):
        del cfg.algorithm.name
    key = (name, family)
    return register[key](cfg, env, env_params)
