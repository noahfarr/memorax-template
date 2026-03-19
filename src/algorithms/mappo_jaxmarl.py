import jax
import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate
from memorax.algorithms import MAPPO
from memorax.networks import FeatureExtractor, heads
from memorax.networks.blocks import GLU, GatedResidual, PreNorm, Projection, Stack
from memorax.utils.typing import Array, Carry


class MultiAgentNetwork(nn.Module):
    feature_extractor: nn.Module
    torso: nn.Module
    head: nn.Module

    @nn.compact
    def __call__(
        self,
        observation: Array,
        done: Array,
        action: Array,
        reward: Array,
        masks: Array,
        initial_carry: Array | None = None,
        **kwargs,
    ) -> tuple[Carry, Array]:
        x, embeddings = self.feature_extractor(
            observation, action=action, reward=reward, done=done
        )

        match self.torso(
            x,
            done=done,
            action=action,
            reward=reward,
            initial_carry=initial_carry,
            **embeddings,
            **kwargs,
        ):
            case (carry, x):
                pass
            case x:
                carry = None

        x = self.head(x, action=action, reward=reward, done=done, **kwargs)
        return carry, x

    @nn.nowrap
    def initialize_carry(self, input_shape: tuple) -> Carry:
        key = jax.random.key(0)
        return getattr(self.torso, "initialize_carry", lambda k, s: None)(
            key, input_shape
        )


def make(cfg, env, env_params):

    action_space = env.action_spaces[env.agents[0]]
    action_dim = action_space.n

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

    actor_network = MultiAgentNetwork(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.Categorical(
            action_dim=action_dim,
        ),
    )

    critic_network = MultiAgentNetwork(
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

    agent = MAPPO(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
        actor_optimizer=optimizer,
        critic_optimizer=optimizer,
    )
    return agent
