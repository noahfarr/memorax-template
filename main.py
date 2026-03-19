import time
from datetime import datetime
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import lox
import orbax.checkpoint as ocp
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from memorax.loggers import MultiLogger
from omegaconf import OmegaConf

from src import algorithm, environment


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):

    env, env_params = environment.make(**cfg.environment)

    algo_num_steps = cfg.algorithm.get(
        "num_steps", cfg.algorithm.get("train_frequency", 128)
    )
    max_episode_steps = (
        env_params.max_steps_in_episode if env_params is not None else algo_num_steps
    )
    num_steps = max(
        cfg.total_timesteps // cfg.num_epochs,
        max(algo_num_steps, max_episode_steps) * cfg.algorithm.num_envs,
    )

    loggers = [
        instantiate(
            v,
            cfg=OmegaConf.to_container(cfg, resolve=True),
            _recursive_=False,
            _convert_="all",
        )
        for v in (cfg.loggers or {}).values()
    ]
    logger = MultiLogger(loggers)

    agent = algorithm.make(cfg, env, env_params)

    key = jax.random.key(cfg.seed)

    init = jax.vmap(agent.init)
    train = jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None))

    key, init_key = jax.random.split(key)
    state = init(jax.random.split(init_key, cfg.num_seeds))

    for _ in range(0, cfg.total_timesteps, num_steps):

        start = time.perf_counter()
        key, train_key = jax.random.split(key)
        state, logs = train(
            jax.random.split(train_key, cfg.num_seeds), state, num_steps
        )
        jax.block_until_ready(state)
        end = time.perf_counter()

        SPS = int(num_steps / (end - start))

        info = logs.pop("info")
        mask = info["returned_episode"]
        episode_returns = jnp.mean(
            info["returned_episode_returns"], where=mask, axis=(1, 2)
        )
        episode_lengths = jnp.mean(
            info["returned_episode_lengths"], where=mask, axis=(1, 2)
        )

        losses = jax.vmap(lambda loss: jax.tree.map(jnp.mean, loss))(logs)

        data = {
            "training/SPS": SPS,
            "training/episode_returns": episode_returns,
            "training/episode_lengths": episode_lengths,
            **losses,
        }
        data = {
            "/".join(str(p.key) for p in path): v
            for path, v in jax.tree_util.tree_leaves_with_path(data)
        }
        logger.log(data, step=state.step.mean(dtype=jnp.int32).item())

    if cfg.checkpoint:
        choices = HydraConfig.get().runtime.choices
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with ocp.StandardCheckpointer() as ckptr:
            for i in range(cfg.num_seeds):
                seed_state = jax.tree.map(lambda x: x[i], state)
                directory = (
                    Path("checkpoints")
                    / cfg.environment.namespace
                    / cfg.environment.env_id
                    / choices["algorithm"]
                    / choices["torso"]
                    / str(i)
                    / timestamp
                ).resolve()
                ckptr.save(directory, seed_state)
            ckptr.wait_until_finished()

    logger.finish()

    return episode_returns.mean().item()


if __name__ == "__main__":
    main()
