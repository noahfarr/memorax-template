from hydra.utils import instantiate
from memorax.algorithms import PQN
from memorax.networks import Network


def make(cfg, env, env_params):
    feature_extractor = instantiate(cfg.feature_extractor)
    torso = instantiate(cfg.stack)
    head = instantiate(cfg.head)

    network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=head,
    )

    agent = PQN(
        cfg=instantiate(cfg.algorithm),
        env=env,
        env_params=env_params,
        q_network=network,
        optimizer=instantiate(cfg.optimizer),
        epsilon_schedule=instantiate(cfg.epsilon_schedule),
    )
    return agent
