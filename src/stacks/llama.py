import flax.linen as nn
from hydra.utils import instantiate
from memorax.networks.blocks import GLU, GatedResidual, PreNorm, Projection, Stack


def llama(features, num_layers, expansion_factor, torso):
    blocks = [Projection(features=features)]
    for _ in range(num_layers):
        blocks.extend(
            [
                GatedResidual(
                    module=PreNorm(norm=nn.RMSNorm, module=instantiate(torso))
                ),
                GatedResidual(
                    module=PreNorm(
                        norm=nn.RMSNorm,
                        module=GLU(
                            features=features,
                            expansion_factor=expansion_factor,
                            activation=nn.silu,
                        ),
                    )
                ),
            ]
        )
    return Stack(blocks=tuple(blocks))
