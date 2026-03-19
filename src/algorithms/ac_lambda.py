from hydra.utils import instantiate
from memorax.algorithms import ACLambda
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

    agent = ACLambda(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
    )
    return agent
