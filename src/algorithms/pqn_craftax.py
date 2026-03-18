from functools import partial

import flax.linen as nn
import jax
import optax
from hydra.utils import instantiate
from memorax.algorithms import PQN
from memorax.networks import Network, heads
from memorax.networks.blocks import GLU, GatedResidual, PreNorm, Projection, Stack

from salt.networks import SelectiveFeatureExtractor


def make(cfg, env, env_params):

    def instantiate_torso_block():
        block = instantiate(cfg.torso)
        if hasattr(block, "features"):
            block = block.clone(features=256)
        return block

    feature_extractor = SelectiveFeatureExtractor(
        observation_extractor=nn.Sequential(
            (
                nn.Dense(features=256),
                nn.LayerNorm(),
                nn.relu,
            )
        ),
        action_extractor=nn.Sequential(
            (partial(jax.nn.one_hot, num_classes=env.action_space(env_params).n),)
        ),
        embeddings=cfg.embeddings,
    )

    blocks = [Projection(features=256)]
    blocks += [
        m
        for _ in range(2)
        for m in (
            GatedResidual(module=PreNorm(norm=nn.RMSNorm, module=instantiate_torso_block())),
            GatedResidual(module=PreNorm(norm=nn.RMSNorm, module=GLU(features=256, expansion_factor=2, activation=nn.silu))),
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

    num_updates_decay = (
        cfg.total_timesteps // cfg.algorithm.num_steps // cfg.algorithm.num_envs
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=3e-4, eps=1e-5),
    )
    epsilon_schedule = optax.linear_schedule(
        init_value=1.0,
        end_value=0.005,
        transition_steps=0.1
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
