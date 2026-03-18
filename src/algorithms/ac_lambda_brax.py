import flax.linen as nn
import jax.numpy as jnp
from hydra.utils import instantiate
from memorax.algorithms import ACLambda
from memorax.networks import FeatureExtractor, Network, heads
from memorax.networks.blocks import GLU, GatedResidual, PreNorm, Projection, Stack


def make(cfg, env, env_params):

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
        head=heads.Gaussian(
            action_dim=env.action_space(env_params).shape[0],
            kernel_init=nn.initializers.orthogonal(0.01),
        ),
    )

    critic_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.VNetwork(
            kernel_init=nn.initializers.orthogonal(1.0),
        ),
    )

    agent = ACLambda(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
    )
    return agent
