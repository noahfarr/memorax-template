from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from hydra.utils import instantiate
from memorax.algorithms import PQN
from memorax.networks import (
    Network,
    heads,
)
from memorax.networks.blocks import GLU, GatedResidual, PreNorm, Projection, Stack

from salt.networks import SelectiveFeatureExtractor

def make(cfg, env, env_params):

    feature_extractor = SelectiveFeatureExtractor(
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
        embeddings=cfg.embeddings,
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

    decay_timesteps = 2_000_000
    num_updates_decay = decay_timesteps // cfg.algorithm.num_steps // cfg.algorithm.num_envs
    num_updates = cfg.total_timesteps // cfg.algorithm.num_steps // cfg.algorithm.num_envs
    eps_decay = 0.25

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.radam(learning_rate=optax.linear_schedule(
            init_value=5e-5,
            end_value=5e-7,
            transition_steps=num_updates * cfg.algorithm.num_minibatches * cfg.algorithm.update_epochs,
        )),
    )
    epsilon_schedule = optax.linear_schedule(
        init_value=1.0,
        end_value=0.05,
        transition_steps=eps_decay * num_updates_decay * cfg.algorithm.num_minibatches * cfg.algorithm.num_steps * cfg.algorithm.num_envs,
    )

    agent = PQN(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        q_network=network,
        optimizer=optimizer,
        epsilon_schedule=epsilon_schedule,
    )
    return agent
