import jax.numpy as jnp


def normalize(x, scale=255.0):
    return x.astype(jnp.float32) / scale
