import flax.linen as nn
import optax
from hydra.utils import instantiate
from memorax.algorithms.ppo import PPO
from memorax.networks import FeatureExtractor, Flatten, Network, heads
from memorax.networks.blocks import GLU, GatedResidual, PreNorm, Projection, Stack


def make(cfg, env, env_params):
    action_dim = env.action_space(env_params).n

    feature_extractor = FeatureExtractor(
        observation_extractor=nn.Sequential(
            (
                nn.Conv(16, (3, 3), strides=1),
                nn.relu,
                Flatten(start_dim=2),
            )
        ),
        action_extractor=nn.Sequential(
            (
                nn.Embed(
                    num_embeddings=action_dim,
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

    actor_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.Categorical(action_dim=action_dim),
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
        optax.adam(learning_rate=2.5e-4, eps=1e-5),
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
