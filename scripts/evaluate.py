import argparse
import os
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from memorax.networks import Network, heads
from memorax.networks.blocks import FFN, GatedResidual, PreNorm, Stack

import salt.utils.resolvers  # noqa: F401 — registers OmegaConf resolvers
from salt.environments.environment import make as make_env
from salt.networks import SelectiveFeatureExtractor

# ── Constants ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = str((ROOT / "config").resolve())

flatten = lambda x: x.reshape(*x.shape[:2], -1).astype(jnp.float32)


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation episodes and save data to parquet."
    )
    parser.add_argument(
        "--environment",
        required=True,
        help="Hydra environment config group (e.g. grimax/dead_reckoning)",
    )
    parser.add_argument(
        "--torso", required=True, help="Hydra torso config group (e.g. salt, gru)"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=20, help="Number of episodes to run"
    )
    parser.add_argument(
        "--checkpoint-seed",
        type=int,
        default=0,
        help="Checkpoint seed directory to load",
    )
    parser.add_argument(
        "--eval-seed", type=int, default=42, help="Base seed for episode RNG"
    )
    parser.add_argument("overrides", nargs="*", help="Extra Hydra overrides")
    return parser.parse_args()


# ── Network building ──────────────────────────────────────────────────────


def build_network(cfg, env, env_params):
    num_actions = env.action_space(env_params).n
    blocks = [
        m
        for _ in range(2)
        for m in (
            GatedResidual(module=PreNorm(module=instantiate(cfg.torso))),
            GatedResidual(module=PreNorm(module=FFN(features=256, expansion_factor=4))),
        )
    ]
    return Network(
        feature_extractor=SelectiveFeatureExtractor(
            observation_extractor=nn.Sequential(
                [
                    flatten,
                    nn.Dense(features=256),
                    nn.relu,
                    nn.LayerNorm(),
                ]
            ),
            action_extractor=partial(jax.nn.one_hot, num_classes=num_actions),
            embeddings=cfg.embeddings,
            features=256,
        ),
        torso=Stack(blocks=tuple(blocks)),
        head=heads.DiscreteQNetwork(action_dim=num_actions),
    )


# ── Checkpoint loading ────────────────────────────────────────────────────


def load_params(cfg, torso_name, seed=0):
    ckpt_root = (
        ROOT
        / "checkpoints"
        / cfg.environment.namespace
        / cfg.environment.env_id
        / "pqn"
        / torso_name
        / str(seed)
    )
    ckpt_path = sorted(ckpt_root.iterdir())[-1]
    print(f"  Loading {torso_name} seed {seed} from {ckpt_path}")
    return ocp.StandardCheckpointer().restore(str(ckpt_path))["params"]


# ── Step function ─────────────────────────────────────────────────────────


def make_step_fn(q_network):
    @jax.jit
    def step_fn(params, obs, prev_action, prev_done, prev_reward, carry):
        obs_seq = obs[None, None]
        action_seq = prev_action[None, None]
        done_seq = prev_done[None, None]
        reward_seq = prev_reward[None, None, None]
        (new_carry, (q_values, _)), state = q_network.apply(
            params,
            observation=obs_seq,
            mask=done_seq,
            action=action_seq,
            reward=reward_seq,
            done=done_seq,
            initial_carry=carry,
            capture_intermediates=True,
            mutable=["intermediates"],
        )
        return new_carry, q_values[0, 0], state

    return step_fn


# ── Hidden state extraction ──────────────────────────────────────────────

# Stack carry is a 4-tuple: (block_0, block_1, block_2, block_3).
# Indices 0 and 2 are the recurrent blocks; 1 and 3 are FFN (carry=None).
# Memoroid blocks: carry[i] = (A_matrix, B_matrix), hidden state is B.
# RNN blocks (GRU): carry[i] = array (the hidden state directly).

RECURRENT_BLOCK_INDICES = [0, 2]


def extract_hidden_state(carry):
    """Return hidden state as a flat numpy array (concatenated over layers)."""
    arrays = []
    for i in RECURRENT_BLOCK_INDICES:
        c = carry[i]
        if c is None:
            continue
        if isinstance(c, tuple):
            # Memoroid: (A, B) — hidden state is B
            arrays.append(np.array(c[1]).ravel())
        else:
            # RNN/GRU: carry is the hidden state array
            arrays.append(np.array(c).ravel())
    return np.concatenate(arrays)


def extract_effective_rank(carry):
    """Mean effective rank of S_t (exp of singular value entropy), over layers/heads.
    Returns NaN for non-matrix carries (GRU, etc.)."""
    try:
        ranks = []
        for i in RECURRENT_BLOCK_INDICES:
            _, S = carry[i]
            for h in range(S.shape[2]):
                sigma = np.linalg.svd(np.array(S[0, 0, h]), compute_uv=False)
                total = sigma.sum()
                if total < 1e-10:
                    ranks.append(1.0)
                    continue
                p = sigma / total
                p = p[p > 1e-10]
                entropy = -np.sum(p * np.log(p))
                ranks.append(float(np.exp(entropy)))
        return np.mean(ranks)
    except (TypeError, ValueError):
        return float("nan")


def extract_innovation_norm(state):
    """Extract mean ||b_t||_F from captured intermediates. Returns NaN if unavailable."""
    try:
        intermediates = state["intermediates"]
        all_norms = []
        for block_key in ["blocks_0", "blocks_2"]:
            B = intermediates["torso"][block_key]["module"]["module"]["cell"][
                "__call__"
            ][0][1]
            per_head = jnp.sqrt(jnp.sum(B[0, 0] ** 2, axis=(-2, -1)))
            all_norms.append(per_head)
        return float(jnp.mean(jnp.concatenate(all_norms)))
    except (KeyError, IndexError):
        return float("nan")


# ── Episode runner ────────────────────────────────────────────────────────


def run_episode(
    step_fn, params, init_carry, env, env_params, seed, horizon, on_step=None
):
    """Run a single episode, returning trajectory data.

    Args:
        on_step: callback(t, carry, *extras) -> dict of diagnostics
    Returns:
        (goal_distance, success, trajectory) where trajectory is a list of dicts.
    """
    key = jax.random.key(seed)
    obs, env_state = env.reset(key, env_params)

    # Extract goal distance if available (grimax-specific)
    try:
        inner = env_state.env_state
        goal_dist = int(jnp.sum(jnp.abs(inner.goal_position - inner.start_position)))
    except AttributeError:
        goal_dist = -1

    carry = init_carry
    prev_action = jnp.array(0, dtype=jnp.int32)
    prev_done = jnp.array(True, dtype=jnp.bool_)
    prev_reward = jnp.array(0.0, dtype=jnp.float32)
    trajectory = []

    for t in range(horizon):
        result = step_fn(params, obs, prev_action, prev_done, prev_reward, carry)
        carry, q_values = result[0], result[1]

        diagnostics = on_step(t, carry, *result[2:]) if on_step else {}

        action = jnp.argmax(q_values)
        key = jax.random.key(seed * 1000 + t)
        obs, env_state, reward, done, info = env.step(
            key, env_state, action, env_params
        )

        trajectory.append(
            {
                "timestep": t,
                "action": int(action),
                "reward": float(reward),
                "done": bool(done),
                **diagnostics,
            }
        )

        prev_action, prev_done, prev_reward = action, done, reward
        if done:
            break

    success = (
        bool(info.get("goal_reached", False))
        if isinstance(info, dict)
        else bool(info["goal_reached"])
    )
    return goal_dist, success, trajectory


# ── Output path ───────────────────────────────────────────────────────────


def make_data_dir(cfg, torso_name):
    """Derive data output directory from environment config and torso name."""
    env_id = cfg.environment.env_id.split("-")[0]
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", env_id).lower()
    data_dir = ROOT / "data" / cfg.environment.namespace / name / torso_name
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            "config",
            overrides=[
                "algorithm=pqn",
                f"environment={args.environment}",
                f"torso={args.torso}",
                *args.overrides,
            ],
        )
        horizon = cfg.environment.kwargs.horizon

        print("Setting up environment...")
        env, env_params = make_env(**cfg.environment)

        net = build_network(cfg, env, env_params)
        params = load_params(cfg, args.torso, seed=args.checkpoint_seed)
        step_fn = make_step_fn(net)
        init_carry = net.initialize_carry((1, None))

        # JIT warmup
        _obs, _ = env.reset(jax.random.key(0), env_params)
        step_fn(
            params,
            _obs,
            jnp.array(0, jnp.int32),
            jnp.array(True, jnp.bool_),
            jnp.array(0.0),
            init_carry,
        )
        print("  JIT warmup done")

        episode_rows = []
        step_rows = []
        prev_state = [np.zeros(1)]

        def on_step(t, carry, state):
            state_vec = extract_hidden_state(carry)
            d = {
                "hidden_state_norm": float(np.linalg.norm(state_vec)),
                "delta_s_norm": float(np.linalg.norm(state_vec - prev_state[0])),
                "effective_rank": extract_effective_rank(carry),
                "innovation_norm": extract_innovation_norm(state),
            }
            prev_state[0] = state_vec
            return d

        for i in range(args.n_episodes):
            ep_seed = args.eval_seed + i
            prev_state[0] = np.zeros(1)

            goal_dist, success, trajectory = run_episode(
                step_fn,
                params,
                init_carry,
                env,
                env_params,
                ep_seed,
                horizon,
                on_step=on_step,
            )

            episode_rows.append(
                {
                    "episode": i,
                    "seed": ep_seed,
                    "goal_distance": goal_dist,
                    "success": success,
                    "num_steps": len(trajectory),
                }
            )

            for row in trajectory:
                row["episode"] = i
                step_rows.append(row)

            status = "OK" if success else "FAIL"
            print(
                f"  Episode {i}: seed={ep_seed}, dist={goal_dist}, "
                f"steps={len(trajectory)}, success={success} [{status}]"
            )

        # Write parquet
        data_dir = make_data_dir(cfg, args.torso)
        episodes_path = data_dir / "episodes.parquet"
        steps_path = data_dir / "steps.parquet"

        pd.DataFrame(episode_rows).to_parquet(episodes_path, index=False)
        pd.DataFrame(step_rows).to_parquet(steps_path, index=False)

        print(f"\nWrote {len(episode_rows)} episodes to {episodes_path}")
        print(f"Wrote {len(step_rows)} steps to {steps_path}")

        # Summary
        df_ep = pd.DataFrame(episode_rows)
        print(
            f"\nSuccess rate: {df_ep['success'].mean():.1%} "
            f"({df_ep['success'].sum()}/{len(df_ep)})"
        )
        print(f"Mean steps: {df_ep['num_steps'].mean():.1f}")


if __name__ == "__main__":
    main()
