from omegaconf import open_dict

from src.algorithms import (
    ac_lambda_brax,
    ac_lambda_popjym,
    dqn_minatar,
    dqn_popgym_arcade,
    dqn_popjym,
    gradient_ppo_brax,
    mappo_jaxmarl,
    ppo_brax,
    ppo_minatar,
    ppo_mujoco_playground,
    ppo_pobax,
    pqn_craftax,
    pqn_grimax,
    pqn_gymnax,
    pqn_gymnasium,
    pqn_minatar,
    pqn_popgym_arcade,
    pqn_popjym,
    pqn_xminigrid,
    r2d2_minatar,
    sac_brax,
    sac_mujoco_playground,
)

register = {
    ("ppo", "brax"): ppo_brax.make,
    ("ppo", "pobax"): ppo_pobax.make,
    ("ppo", "minatar"): ppo_minatar.make,
    ("ppo", "mujoco_playground"): ppo_mujoco_playground.make,
    ("pqn", "gymnax"): pqn_gymnax.make,
    ("pqn", "grimax"): pqn_grimax.make,
    ("pqn", "minatar"): pqn_minatar.make,
    ("pqn", "popjym"): pqn_popjym.make,
    ("pqn", "popgym_arcade"): pqn_popgym_arcade.make,
    ("pqn", "craftax"): pqn_craftax.make,
    ("pqn", "navix"): pqn_grimax.make,
    ("pqn", "xminigrid"): pqn_xminigrid.make,
    ("pqn", "gymnasium"): pqn_gymnasium.make,
    ("dqn", "minatar"): dqn_minatar.make,
    ("dqn", "popjym"): dqn_popjym.make,
    ("dqn", "popgym_arcade"): dqn_popgym_arcade.make,
    ("sac", "brax"): sac_brax.make,
    ("sac", "mujoco_playground"): sac_mujoco_playground.make,
    ("ac_lambda", "popjym"): ac_lambda_popjym.make,
    ("ac_lambda", "brax"): ac_lambda_brax.make,
    ("gradient_ppo", "brax"): gradient_ppo_brax.make,
    ("mappo", "jaxmarl"): mappo_jaxmarl.make,
    ("r2d2", "minatar"): r2d2_minatar.make,
}


def make(cfg, env, env_params):
    suite = cfg.environment.get("suite", cfg.environment.namespace)
    name = cfg.algorithm.name
    with open_dict(cfg):
        del cfg.algorithm.name
        del cfg.torso.name
    key = (name, suite)
    return register[key](cfg, env, env_params)
