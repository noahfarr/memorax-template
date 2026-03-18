from datetime import datetime
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src import algorithm, environment


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):

    env, env_params = environment.make(**cfg.environment)

    num_train_steps = (
        max(cfg.algorithm.num_steps, env_params.max_steps_in_episode)
        * cfg.algorithm.num_envs
    )

    logger = instantiate(cfg.logger)
    logger_state = logger.init(cfg=OmegaConf.to_container(cfg, resolve=True))

    agent = algorithm.make(cfg, env, env_params)

    key = jax.random.key(cfg.seed)
    keys = jax.random.split(key, cfg.num_seeds)

    init = jax.vmap(agent.init)
    train = jax.vmap(agent.train, in_axes=(0, 0, None))

    if cfg.num_eval_steps:
        evaluate = jax.vmap(agent.evaluate, in_axes=(0, 0, None))

    def episode_stats(t):
        mask = t.metadata["returned_episode"]
        returns = jnp.where(mask, t.metadata["returned_episode_returns"], jnp.nan)
        lengths = jnp.where(mask, t.metadata["returned_episode_lengths"], jnp.nan)
        return {
            "mean_episode_returns": jnp.nanmean(returns),
            "mean_episode_lengths": jnp.nanmean(lengths),
            "num_episodes": jnp.sum(mask),
        }

    def log(state, transitions, SPS, logger_state, prefix):
        infos = jax.vmap(lambda t: t.info)(transitions)
        leaves, treedef = jax.tree.flatten_with_path(infos)
        keys = [
            jax.tree_util.keystr(path).replace("']['", "/").strip("['']")
            for path, _ in leaves
        ]
        infos = {f"{prefix}/{k}": v for k, (_, v) in zip(keys, leaves)}
        losses = jax.vmap(lambda t: t.losses)(transitions)
        ep_stats = jax.vmap(episode_stats)(transitions)
        ep_stats = {f"{prefix}/{k}": v for k, v in ep_stats.items()}
        ep_stats = {k: jnp.where(jnp.isnan(v), 0.0, v) for k, v in ep_stats.items()}
        data = {**infos, **losses, **ep_stats, f"{prefix}/SPS": SPS}
        logger_state = logger.log(logger_state, data, step=state.step[0].item())
        logger.emit(logger_state)
        return logger_state

    keys, state = init(keys)

    for step in range(0, cfg.total_timesteps, num_train_steps):

        (keys, state, transitions), SPS = train(keys, state, num_train_steps)

        logger_state = log(state, transitions, SPS, logger_state, "training")

        if cfg.num_eval_steps:
            (keys, transitions), SPS = evaluate(keys, state, cfg.num_eval_steps)
            logger_state = log(
                state,
                transitions,
                SPS,
                logger_state,
                "evaluation",
            )

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

    logger.finish(logger_state)


if __name__ == "__main__":
    main()
