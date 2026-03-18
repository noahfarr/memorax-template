from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import flashbax
from hydra.utils import instantiate
from memorax.algorithms import DQN
from memorax.networks import (
    FeatureExtractor,
    Network,
    heads,
)
from memorax.networks.blocks import GLU, GatedResidual, PreNorm, Projection, Stack


def make(cfg, env, env_params):

    feature_extractor = FeatureExtractor(
        observation_extractor=nn.Sequential([
            lambda x: x.astype(jnp.float32) / 255.0,
            nn.Conv(64, (5, 5), strides=2, padding='VALID'),
            nn.leaky_relu, partial(nn.max_pool, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(128, (3, 3), strides=2, padding='VALID'),
            nn.leaky_relu, partial(nn.max_pool, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(256, (3, 3), strides=2, padding='VALID'),
            nn.leaky_relu, partial(nn.max_pool, window_shape=(3, 3), strides=(1, 1)),
            nn.Conv(512, (1, 1), strides=1, padding='VALID'),
            nn.leaky_relu,
            lambda x: x.reshape(*x.shape[:2], -1).astype(jnp.float32),
        ]),
        action_extractor=lambda x: jax.nn.one_hot(x, env.action_space(env_params).n),
    )

    blocks = [Projection(features=256)]
    blocks += [m for _ in range(2) for m in (
        GatedResidual(module=PreNorm(norm=nn.RMSNorm, module=instantiate(cfg.torso))),
        GatedResidual(module=PreNorm(norm=nn.RMSNorm, module=GLU(features=256, expansion_factor=4, activation=nn.silu))),
    )]
    torso = Stack(blocks=tuple(blocks))

    head = heads.DiscreteQNetwork(
        action_dim=env.action_space(env_params).n,
    )

    network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=head,
    )

    num_updates = cfg.total_timesteps // cfg.algorithm.num_steps // cfg.algorithm.num_envs

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.radam(learning_rate=optax.linear_schedule(
            init_value=5e-5,
            end_value=5e-7,
            transition_steps=num_updates,
        )),
    )
    epsilon_schedule = optax.linear_schedule(
        init_value=1.0,
        end_value=0.05,
        transition_steps=int(0.25 * cfg.total_timesteps),
    )

    buffer = flashbax.make_trajectory_buffer(
        add_batch_size=cfg.algorithm.num_envs,
        sample_batch_size=32,
        sample_sequence_length=cfg.algorithm.num_steps,
        period=1,
        min_length_time_axis=cfg.algorithm.num_steps,
        max_length_time_axis=10_000,
    )

    agent = DQN(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        q_network=network,
        optimizer=optimizer,
        buffer=buffer,
        epsilon_schedule=epsilon_schedule,
    )
    return agent
