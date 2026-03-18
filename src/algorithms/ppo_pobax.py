import flax.linen as nn
import optax
from hydra.utils import instantiate
from memorax.algorithms.ppo import PPO
from memorax.networks import (
    FeatureExtractor,
    Network,
    heads,
)
from memorax.networks.blocks import GLU, GatedResidual, PreNorm, Projection, Stack


def make(cfg, env, env_params):
    action_space = env.action_space(env_params)
    action_dim = getattr(action_space, "n", None) or action_space.shape[0]
    hidden_size = cfg.get("hidden_size", 256)

    observation_extractor = instantiate(cfg.observation_extractor)
    action_extractor = instantiate(cfg.action_extractor, action_dim=action_dim)
    head = instantiate(cfg.head, action_dim=action_dim)

    feature_extractor = FeatureExtractor(
        observation_extractor=observation_extractor,
        action_extractor=action_extractor,
    )

    blocks = [Projection(features=hidden_size)]
    blocks += [m for _ in range(2) for m in (
        GatedResidual(module=PreNorm(norm=nn.RMSNorm, module=instantiate(cfg.torso))),
        GatedResidual(module=PreNorm(norm=nn.RMSNorm, module=GLU(features=hidden_size, expansion_factor=4, activation=nn.silu))),
    )]
    torso = Stack(blocks=tuple(blocks))

    actor_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=head,
    )

    critic_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.VNetwork(
            kernel_init=nn.initializers.orthogonal(1.0),
        ),
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=3e-4),
    )

    agent = PPO(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
        actor_optimizer=optimizer,
        critic_optimizer=optimizer,
    )
    return agent
