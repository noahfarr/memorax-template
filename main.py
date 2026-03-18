import time
from datetime import datetime
from pathlib import Path

import hydra
import jax
import lox
import orbax.checkpoint as ocp
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from memorax.loggers import MultiLogger

from src import algorithm, environment


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):

    env, env_params = environment.make(**cfg.environment)

    num_train_steps = (
        max(cfg.algorithm.num_steps, env_params.max_steps_in_episode)
        * cfg.algorithm.num_envs
    )

    loggers = [instantiate(v) for v in (cfg.loggers or {}).values()]
    logger = MultiLogger(loggers)

    agent = algorithm.make(cfg, env, env_params)

    key = jax.random.key(cfg.seed)

    init = jax.vmap(agent.init)
    train = jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None))

    if cfg.num_eval_steps:
        evaluate = jax.vmap(lox.spool(agent.evaluate), in_axes=(0, 0, None))

    key, init_key = jax.random.split(key)
    state = init(jax.random.split(init_key, cfg.num_seeds))

    for step in range(0, cfg.total_timesteps, num_train_steps):

        start = time.perf_counter()
        key, train_key = jax.random.split(key)
        state, logs = train(
            jax.random.split(train_key, cfg.num_seeds), state, num_train_steps
        )
        jax.block_until_ready(state)
        end = time.perf_counter()

        SPS = int(num_train_steps / (end - start))

        info = logs.pop("info")
        episode_returns = info["returned_episode_returns"][info["returned_episode"]]
        episode_lengths = info["returned_episode_lengths"][info["returned_episode"]]

        data = {
            "training/SPS": SPS,
            "training/episode_returns": episode_returns,
            "training/episode_lengths": episode_lengths,
            **logs,
        }
        logger.log(data, step=state.step.mean().item())

        if cfg.num_eval_steps:
            key, eval_key = jax.random.split(key)
            state, eval_logs = evaluate(
                jax.random.split(eval_key, cfg.num_seeds), state, cfg.num_eval_steps
            )

            eval_info = eval_logs.pop("info")
            eval_returns = eval_info["returned_episode_returns"][
                eval_info["returned_episode"]
            ]
            eval_lengths = eval_info["returned_episode_lengths"][
                eval_info["returned_episode"]
            ]

            eval_data = {
                "evaluation/episode_returns": eval_returns,
                "evaluation/episode_lengths": eval_lengths,
                **eval_logs,
            }
            logger.log(eval_data, step=state.step.mean().item())

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


if __name__ == "__main__":
    main()
