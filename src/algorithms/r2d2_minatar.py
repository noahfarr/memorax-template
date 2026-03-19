import flax.linen as nn
import jax.numpy as jnp
import optax
import flashbax
from hydra.utils import instantiate
from memorax.algorithms import R2D2
from memorax.networks import FeatureExtractor, Network, heads
from memorax.networks.blocks import GLU, GatedResidual, PreNorm, Projection, Stack
from omegaconf import open_dict

flatten = lambda x: x.reshape(*x.shape[:2], -1).astype(jnp.float32)


def make(cfg, env, env_params):

    # R2D2Config does not have num_steps; remove before instantiation
    with open_dict(cfg):
        num_steps = cfg.algorithm.pop("num_steps")

    feature_extractor = FeatureExtractor(
        observation_extractor=nn.Sequential(
            (
                nn.Conv(16, (3, 3), strides=1),
                nn.relu,
                flatten,
            )
        ),
        action_extractor=nn.Sequential(
            (
                nn.Embed(
                    num_embeddings=env.action_space(env_params).n,
                    features=32,
                ),
                nn.relu,
                nn.LayerNorm(),
            )
        ),
    )

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
        optax.adam(learning_rate=5e-4, eps=1e-5),
    )
    epsilon_schedule = optax.linear_schedule(
        init_value=1.0,
        end_value=0.05,
        transition_steps=int(0.2 * cfg.total_timesteps),
    )
    beta_schedule = optax.linear_schedule(
        init_value=cfg.algorithm.importance_sampling_exponent,
        end_value=1.0,
        transition_steps=cfg.total_timesteps,
    )

    buffer = flashbax.make_prioritised_trajectory_buffer(
        add_batch_size=cfg.algorithm.num_envs,
        sample_batch_size=32,
        sample_sequence_length=cfg.algorithm.sequence_length + cfg.algorithm.burn_in_length,
        period=1,
        min_length_time_axis=cfg.algorithm.sequence_length + cfg.algorithm.burn_in_length,
        max_length_time_axis=10_000,
        priority_exponent=cfg.algorithm.priority_exponent,
    )

    agent = R2D2(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        q_network=network,
        optimizer=optimizer,
        buffer=buffer,
        epsilon_schedule=epsilon_schedule,
        beta_schedule=beta_schedule,
    )
    return agent
