from pathlib import Path

import gymnax
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src import environment


def get_action_dim(cfg):
    env, env_params = environment.make(**cfg)

    if isinstance(env.action_space(env_params), gymnax.environments.spaces.Discrete):
        action_dim = env.action_space(env_params).n
    else:

        action_dim, *_ = env.action_space(env_params).shape
    return action_dim


def cascading_fallback(algorithm: str, environment: str, torso=None) -> str:
    gh = GlobalHydra.instance()

    loader = gh.config_loader()

    env_path = Path(environment)

    # Try most specific to least specific path.
    # e.g. for "gymnax/minatar/breakout": try breakout, then minatar, then gymnax
    for path in [env_path, *env_path.parents]:
        if path == Path("."):
            break
        search_group = f"hyperparameters/{algorithm}/{path.parent}" if str(path.parent) != "." else f"hyperparameters/{algorithm}"
        if path.name in loader.get_group_options(search_group):
            result = f"{algorithm}/{path}"
            if torso and torso in loader.get_group_options(f"hyperparameters/{result}"):
                return f"{result}/{torso}"
            return result

    return f"{algorithm}/{env_path.parts[0]}"


def get_group(_root_):
    group = f"{_root_.algorithm.name}_{_root_.environment.namespace}_{_root_.environment.env_id}_{_root_.torso.name}_{'_'.join(f'{k}_{v}' for k, v in sorted(_root_.environment.get('kwargs', {}).items()))}"
    if len(group) > 128:
        group = group[:128]
    return group


OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("get_action_dim", get_action_dim)
OmegaConf.register_new_resolver("cascading_fallback", cascading_fallback)
OmegaConf.register_new_resolver("get_group", get_group)
