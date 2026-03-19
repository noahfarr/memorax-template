import jax
import flax.linen as nn
from hydra.utils import instantiate
from memorax.algorithms import MAPPO
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
    feature_extractor = instantiate(cfg.feature_extractor)
    torso = instantiate(cfg.stack)

    actor_network = MultiAgentNetwork(
        feature_extractor=feature_extractor,
        torso=torso,
        head=instantiate(cfg.actor_head),
    )

    critic_network = MultiAgentNetwork(
        feature_extractor=feature_extractor,
        torso=torso,
        head=instantiate(cfg.critic_head),
    )

    optimizer = instantiate(cfg.optimizer)

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
