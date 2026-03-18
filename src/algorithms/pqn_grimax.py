from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from hydra.utils import instantiate
from memorax.algorithms import PQN
from memorax.networks import Network, heads
from memorax.networks.blocks import GLU, GatedResidual, PreNorm, Stack
from omegaconf import OmegaConf

from salt.networks import SelectiveFeatureExtractor

flatten = lambda x: x.reshape(*x.shape[:2], -1).astype(jnp.float32)


def make(cfg, env, env_params):

    num_actions = env.action_space(env_params).n

    feature_extractor = SelectiveFeatureExtractor(
        observation_extractor=nn.Sequential(
            (
                flatten,
                nn.Dense(features=256),
                nn.relu,
                nn.LayerNorm(),
            )
        ),
        action_extractor=partial(jax.nn.one_hot, num_classes=num_actions),
        embeddings=cfg.embeddings,
        features=256,
    )

    blocks = [
        m
        for _ in range(2)
        for m in (
            GatedResidual(module=PreNorm(norm=nn.RMSNorm, module=instantiate(cfg.torso))),
            GatedResidual(module=PreNorm(norm=nn.RMSNorm, module=GLU(features=256, expansion_factor=4, activation=nn.silu))),
        )
    ]
    torso = Stack(blocks=tuple(blocks))

    head = heads.DiscreteQNetwork(
        action_dim=env.action_space(env_params).n,
    )

    network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=head,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=3e-4, eps=1e-5),
    )
    epsilon_schedule = optax.linear_schedule(
        init_value=1.0,
        end_value=0.05,
        transition_steps=int(0.25 * cfg.total_timesteps),
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
