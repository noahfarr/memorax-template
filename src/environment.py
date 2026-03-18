from hydra.utils import instantiate
from memorax.environments import environment


def make(namespace, env_id, **kwargs):
    env_id = f"{namespace}::{env_id}"
    env, env_params = environment.make(env_id, **(kwargs.get("kwargs") or {}))

    if env_params is not None:
        env_params = env_params.replace(**kwargs.get("env_params", {}))

    for wrapper in kwargs.get("wrappers", []):
        env = instantiate(wrapper, env)

    return env, env_params
