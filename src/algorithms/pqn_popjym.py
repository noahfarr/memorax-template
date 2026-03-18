from functools import partial

import flax.linen as nn
import jax
import optax
from hydra.utils import instantiate
from memorax.algorithms import PQN
from memorax.networks import FeatureExtractor, Network, heads
from memorax.networks.blocks import GLU, GatedResidual, PreNorm, Projection, Stack


def make(cfg, env, env_params):

    feature_extractor = FeatureExtractor(
        observation_extractor=nn.Sequential(
            (nn.Dense(features=256), nn.LayerNorm(), nn.relu)
        ),
        action_extractor=nn.Sequential(
            (partial(jax.nn.one_hot, num_classes=env.action_space(env_params).n),)
        ),
    )

    blocks = [Projection(features=256)]
    blocks += [
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

    decay_timesteps = 2_000_000
    num_updates_decay = (
        decay_timesteps // cfg.algorithm.num_steps // cfg.algorithm.num_envs
    )
    eps_decay = 0.25

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=3e-4, eps=1e-5),
    )
    epsilon_schedule = optax.linear_schedule(
        init_value=1.0,
        end_value=0.05,
        transition_steps=eps_decay
        * num_updates_decay
        * cfg.algorithm.num_minibatches
        * cfg.algorithm.num_steps
        * cfg.algorithm.num_envs,
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
