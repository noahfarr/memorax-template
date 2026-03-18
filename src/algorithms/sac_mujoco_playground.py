import flax.linen as nn
import jax.numpy as jnp
import optax
import flashbax
from hydra.utils import instantiate
from memorax.algorithms.sac import SAC
from memorax.networks import FeatureExtractor, Network, heads
from memorax.networks.blocks import GLU, GatedResidual, PreNorm, Projection, Stack


def make(cfg, env, env_params):

    action_dim = env.action_space(env_params).shape[0]

    blocks = [Projection(features=256)]
    blocks += [
        m
        for _ in range(2)
        for m in (
            GatedResidual(
                module=PreNorm(norm=nn.RMSNorm, module=instantiate(cfg.torso))
            ),
            GatedResidual(
                module=PreNorm(
                    norm=nn.RMSNorm,
                    module=GLU(features=256, expansion_factor=4, activation=nn.silu),
                )
            ),
        )
    ]
    torso = Stack(blocks=tuple(blocks))

    feature_extractor = FeatureExtractor(
        observation_extractor=nn.Sequential(
            [
                nn.Dense(256, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2))),
                nn.tanh,
                nn.Dense(256, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2))),
                nn.tanh,
            ]
        ),
        action_extractor=nn.Sequential(
            [
                nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2))),
                nn.tanh,
            ]
        ),
    )

    actor_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.SquashedGaussian(
            action_dim=action_dim,
        ),
    )

    critic_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.TwinContinuousQNetwork(),
    )

    alpha_network = heads.Alpha(initial_alpha=1.0)

    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=3e-4),
    )
    critic_optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=3e-4),
    )
    alpha_optimizer = optax.adam(learning_rate=3e-4)

    buffer = flashbax.make_trajectory_buffer(
        add_batch_size=cfg.algorithm.num_envs,
        sample_batch_size=256,
        sample_sequence_length=cfg.algorithm.num_steps,
        period=1,
        min_length_time_axis=cfg.algorithm.num_steps,
        max_length_time_axis=100_000,
    )

    agent = SAC(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
        alpha_network=alpha_network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        alpha_optimizer=alpha_optimizer,
        buffer=buffer,
    )
    return agent
