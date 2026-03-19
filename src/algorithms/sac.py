from hydra.utils import instantiate
from memorax.algorithms.sac import SAC
from memorax.networks import Network


def make(cfg, env, env_params):
    feature_extractor = instantiate(cfg.feature_extractor)
    torso = instantiate(cfg.stack)

    actor_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=instantiate(cfg.actor_head),
    )

    critic_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=instantiate(cfg.critic_head),
    )

    agent = SAC(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
        alpha_network=instantiate(cfg.alpha),
        actor_optimizer=instantiate(cfg.actor_optimizer),
        critic_optimizer=instantiate(cfg.critic_optimizer),
        alpha_optimizer=instantiate(cfg.alpha_optimizer),
        buffer=instantiate(cfg.buffer),
    )
    return agent
