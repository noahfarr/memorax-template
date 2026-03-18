import gymnax
import jax
import jax.numpy as jnp
from gymnax.visualize import Visualizer

from salt.environments.environment import make

key = jax.random.key(2)
key, key_reset, key_policy, key_step = jax.random.split(key, 4)

# env, env_params = gymnax.make("Freeway-MinAtar")
env_id = "Freeway"
env, env_params = make("poatar", env_id)

state_seq, reward_seq = [], []
key, key_reset = jax.random.split(key)
obs, env_state = env.reset(key_reset, env_params)

step = jax.jit(env.step)
sample = jax.jit(env.action_space(env_params).sample)

for i in range(100):
    state_seq.append(env_state)
    key, key_act, key_step = jax.random.split(key, 3)
    action = sample(key_act)
    next_obs, next_env_state, reward, done, info = step(
        key_step, env_state, action, env_params
    )
    reward_seq.append(reward)
    if done:
        break
    else:
        obs = next_obs
        env_state = next_env_state

cum_rewards = jnp.cumsum(jnp.array(reward_seq))
vis = Visualizer(env, env_params, state_seq, cum_rewards)
vis.animate(f"videos/poatar_freeway.gif")
