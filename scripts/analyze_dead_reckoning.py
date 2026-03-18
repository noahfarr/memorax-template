"""Dead Reckoning analysis: hidden state norm and performance profile plots.

Loads trained SALT and Gated DeltaNet checkpoints, runs evaluation episodes
on the Dead Reckoning environment, and produces two diagnostic figures:
1. Hidden state Frobenius norm over timesteps (single episode per model)
2. Success rate vs goal distance (1024 parallel episodes, single seed)
"""

import os
import re
import sys
from pathlib import Path

# Ensure project root is on sys.path so `salt` package is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
import seaborn as sns
from fanda.fanda import Fanda
from fanda.utils import save_fig
from fanda.visualizations import add_legend, annotate_axis, decorate_axis
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from memorax.networks import Network, heads
from memorax.networks.blocks import FFN, GatedResidual, PreNorm, Stack

import salt.utils.resolvers  # noqa: F401 — registers OmegaConf resolvers
from salt.environments.environment import make as make_env
from salt.networks import SelectiveFeatureExtractor
from salt.networks.cells.delta_rule import GatedDeltaRuleCell
from salt.networks.cells.salt import SALTCell

# ── Constants ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = str((ROOT / "config").resolve())

TORSOS = ["salt", "gated_delta_net"]
LABELS = {"salt": "SALT", "gated_delta_net": "Gated DeltaNet"}

ENV_OVERRIDES = [
    "environment.kwargs.grid_size=64",
    "environment.kwargs.min_goal_distance=32",
    "environment.kwargs.horizon=200",
]

CELL_CLS = {
    "salt": SALTCell,
    "gated_delta_net": GatedDeltaRuleCell,
}

NUM_ENVS = 1024
SEED = 0
NORM_ENV_SEED = 42  # seed for the single norm-plot episode

flatten = lambda x: x.reshape(*x.shape[:2], -1).astype(jnp.float32)


def make_plot_dir(cfg):
    """Derive plot output directory from environment config."""
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', cfg.environment.env_id.split("-")[0]).lower()
    plot_dir = ROOT / "plots" / cfg.environment.namespace / name
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


# ── Network building ──────────────────────────────────────────────────────


def build_network(cfg, env, env_params):
    num_actions = env.action_space(env_params).n
    blocks = [m for _ in range(2) for m in (
        GatedResidual(module=PreNorm(module=instantiate(cfg.torso))),
        GatedResidual(module=PreNorm(module=FFN(features=256, expansion_factor=4))),
    )]
    return Network(
        feature_extractor=SelectiveFeatureExtractor(
            observation_extractor=nn.Sequential([
                flatten,
                nn.Dense(features=256),
                nn.relu,
                nn.LayerNorm(),
            ]),
            action_extractor=partial(jax.nn.one_hot, num_classes=num_actions),
            embeddings=cfg.embeddings,
            features=256,
        ),
        torso=Stack(blocks=tuple(blocks)),
        head=heads.DiscreteQNetwork(action_dim=num_actions),
    )


# ── Checkpoint loading ─────────────────────────────────────────────────────


def load_params(cfg, torso_name, seed=0):
    ckpt_root = ROOT / "checkpoints" / cfg.environment.namespace / cfg.environment.env_id / "pqn" / torso_name / str(seed)
    ckpt_path = sorted(ckpt_root.iterdir())[-1]
    print(f"  Loading {torso_name} seed {seed} from {ckpt_path}")
    return ocp.StandardCheckpointer().restore(str(ckpt_path))["params"]


# ── Environment helpers ────────────────────────────────────────────────────


def reset_batch(env, env_params, base_seed, num_envs):
    keys = jax.random.split(jax.random.key(base_seed), num_envs)
    obs, env_states = jax.vmap(env.reset, in_axes=(0, None))(keys, env_params)
    return obs, env_states


def step_batch(env, env_params, keys, env_states, actions):
    return jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        keys, env_states, actions, env_params
    )


def get_goal_distances(env_states):
    inner = env_states.env_state
    return jnp.sum(jnp.abs(inner.goal_position - inner.start_position), axis=-1)


# ── JIT'd forward pass ─────────────────────────────────────────────────────


def make_step_fn(q_network, cell_cls):
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
            capture_intermediates=lambda mod, method: isinstance(mod, cell_cls) and method == '__call__',
            mutable=['intermediates'],
        )
        q_values = q_values[0, 0]
        return new_carry, q_values, state
    return step_fn


def make_batch_step_fn(q_network):
    @jax.jit
    def step_fn(params, obs, prev_action, prev_done, prev_reward, carry):
        obs_seq = obs[:, None]
        action_seq = prev_action[:, None]
        done_seq = prev_done[:, None]
        reward_seq = prev_reward[:, None, None]

        (new_carry, (q_values, _)), _ = q_network.apply(
            params,
            observation=obs_seq,
            mask=done_seq,
            action=action_seq,
            reward=reward_seq,
            done=done_seq,
            initial_carry=carry,
            mutable=['intermediates'],
        )
        q_values = q_values[:, 0]
        return new_carry, q_values
    return step_fn


def extract_innovation_norm(state):
    """Extract mean ||b_t||_F (innovation norm) from captured intermediates."""
    intermediates = state['intermediates']
    all_norms = []
    for block_key in ['blocks_0', 'blocks_2']:
        B = intermediates['torso'][block_key]['module']['module']['cell']['__call__'][0][1]
        per_head = jnp.sqrt(jnp.sum(B[0, 0] ** 2, axis=(-2, -1)))
        all_norms.append(per_head)
    return float(jnp.mean(jnp.concatenate(all_norms)))


def extract_effective_rank(carry):
    """Extract mean effective rank of S_t from carry (mean over layers and heads).

    Effective rank = exp(entropy of normalized singular values).

    carry is a tuple of 4 elements (one per Stack block).
    Indices 0 and 2 are the Memoroid carries: each is (A_matrix, B_matrix).
    B_matrix (the hidden state) has shape (1, 1, num_heads, head_dim, head_dim).

    Returns: scalar float.
    """
    ranks = []
    for carry_idx in [0, 2]:
        _, S = carry[carry_idx]
        for h in range(S.shape[2]):
            sigma = np.linalg.svd(np.array(S[0, 0, h]), compute_uv=False)
            # Normalize to probability distribution
            total = sigma.sum()
            if total < 1e-10:
                ranks.append(1.0)
                continue
            p = sigma / total
            p = p[p > 1e-10]  # avoid log(0)
            entropy = -np.sum(p * np.log(p))
            ranks.append(float(np.exp(entropy)))
    return np.mean(ranks)


# ── Episode runner ─────────────────────────────────────────────────────────


def run_episode(step_fn, params, init_carry, env, env_params, seed, horizon,
                on_step=None):
    key = jax.random.key(seed)
    obs, env_state = env.reset(key, env_params)
    inner = env_state.env_state
    goal_dist = int(jnp.sum(jnp.abs(inner.goal_position - inner.start_position)))

    carry = init_carry
    prev_action = jnp.array(0, dtype=jnp.int32)
    prev_done = jnp.array(True, dtype=jnp.bool_)
    prev_reward = jnp.array(0.0, dtype=jnp.float32)

    for t in range(horizon):
        result = step_fn(params, obs, prev_action, prev_done, prev_reward, carry)
        carry, q_values = result[0], result[1]
        if on_step:
            on_step(t, carry, *result[2:])

        action = jnp.argmax(q_values)
        key = jax.random.key(seed * 1000 + t)
        obs, env_state, reward, done, info = env.step(key, env_state, action, env_params)
        prev_action, prev_done, prev_reward = action, done, reward
        if done:
            break

    return goal_dist, bool(info["goal_reached"])


# ── Plot 1: Hidden State Diagnostics (single episode) ─────────────────────


def plot_hidden_state_diagnostics(salt_ranks, salt_innov, delta_ranks, delta_innov,
                                   salt_dist, delta_dist, plot_dir):
    max_len = min(len(salt_ranks), len(delta_ranks)) - 3
    labels = ["SALT", "Gated DeltaNet"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Panel 1: Effective Rank
    df_rank = pd.DataFrame([
        *[{"timestep": t, "effective_rank": v, "network": "SALT"} for t, v in enumerate(salt_ranks[:max_len])],
        *[{"timestep": t, "effective_rank": v, "network": "Gated DeltaNet"} for t, v in enumerate(delta_ranks[:max_len])],
    ])
    sns.lineplot(data=df_rank, x="timestep", y="effective_rank",
                 hue="network", hue_order=labels, palette="colorblind", ax=ax1)
    (
        Fanda(fig=fig, ax=ax1)
        .pipe(annotate_axis, xlabel="Timestep", ylabel="Effective Rank")
        .pipe(decorate_axis, ticklabelsize="xx-large")
        .pipe(add_legend, labels=labels)
    )

    # Panel 2: Innovation Magnitude
    df_innov = pd.DataFrame([
        *[{"timestep": t, "innovation": v, "network": "SALT"} for t, v in enumerate(salt_innov[:max_len])],
        *[{"timestep": t, "innovation": v, "network": "Gated DeltaNet"} for t, v in enumerate(delta_innov[:max_len])],
    ])
    sns.lineplot(data=df_innov, x="timestep", y="innovation",
                 hue="network", hue_order=labels, palette="colorblind", ax=ax2)
    (
        Fanda(fig=fig, ax=ax2)
        .pipe(annotate_axis, xlabel="Timestep", ylabel=r"$\|\mathbf{b}_t\|_F$")
        .pipe(decorate_axis, ticklabelsize="xx-large")
        .pipe(add_legend, labels=labels)
    )

    fig.tight_layout()
    path = str(plot_dir / "hidden_state_diagnostics")
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    print(f"Saved {path}.pdf")
    plt.close(fig)

    # Standalone innovation plot
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(data=df_innov, x="timestep", y="innovation",
                 hue="network", hue_order=labels, palette="colorblind", ax=ax)
    (
        Fanda(fig=fig, ax=ax)
        .pipe(annotate_axis, xlabel="Timestep", ylabel=r"$\|\mathbf{b}_t\|_F$")
        .pipe(decorate_axis, ticklabelsize="xx-large")
        .pipe(add_legend, labels=labels)
        .pipe(save_fig, name=str(plot_dir / "innovation_magnitude"))
    )
    print(f"Saved {plot_dir / 'innovation_magnitude'}.pdf")
    plt.close(fig)


# ── Plot 2: Performance Profile (batched) ─────────────────────────────────


def run_eval_episodes(step_fn, params, init_carry, env, env_params, seed, horizon):
    obs, env_states = reset_batch(env, env_params, seed * 10000, NUM_ENVS)
    goal_dists = get_goal_distances(env_states)

    carry = init_carry
    prev_action = jnp.zeros(NUM_ENVS, dtype=jnp.int32)
    prev_done = jnp.ones(NUM_ENVS, dtype=jnp.bool_)
    prev_reward = jnp.zeros(NUM_ENVS, dtype=jnp.float32)
    ever_reached = jnp.zeros(NUM_ENVS, dtype=jnp.bool_)
    active = jnp.ones(NUM_ENVS, dtype=jnp.bool_)

    for t in range(horizon):
        carry, q_values = step_fn(
            params, obs, prev_action, prev_done, prev_reward, carry
        )
        actions = jnp.argmax(q_values, axis=-1)

        keys = jax.random.split(jax.random.key(seed * 100000 + t), NUM_ENVS)
        obs, env_states, rewards, dones, infos = step_batch(
            env, env_params, keys, env_states, actions
        )

        ever_reached = ever_reached | (active & infos["goal_reached"])
        prev_action = actions
        prev_done = dones
        prev_reward = rewards
        active = active & ~dones

        if not jnp.any(active):
            break

    results = list(zip(
        np.array(goal_dists).tolist(),
        np.array(ever_reached).tolist(),
    ))
    return results


def plot_performance_profile(salt_results, delta_results, plot_dir):
    buckets = [
        ("32-64", 32, 64),
        ("64-128", 64, 128),
    ]

    def bucket_success_rates(results):
        rates = []
        counts = []
        for label, lo, hi in buckets:
            hits = [r for d, r in results if lo <= d < hi]
            if hits:
                rates.append(np.mean(hits))
                counts.append(len(hits))
            else:
                rates.append(0.0)
                counts.append(0)
        return rates, counts

    salt_rates, salt_counts = bucket_success_rates(salt_results)
    delta_rates, delta_counts = bucket_success_rates(delta_results)

    x = np.arange(len(buckets))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    bars1 = ax.bar(x - width / 2, salt_rates, width, label="SALT",
                   color="#2ca02c", alpha=0.85)
    bars2 = ax.bar(x + width / 2, delta_rates, width, label="Gated DeltaNet",
                   color="#d62728", alpha=0.85)

    for bars, counts in [(bars1, salt_counts), (bars2, delta_counts)]:
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"n={count}", ha='center', va='bottom', fontsize=7)

    ax.set_xlabel("Goal Distance (Manhattan)", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Success Rate by Goal Distance", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in buckets])
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = plot_dir / "performance_profile.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        norm_data = {}
        eval_data = {}
        plot_dir = None

        for torso_name in TORSOS:
            cfg = compose("config", overrides=[
                "algorithm=pqn",
                "environment=grimax/dead_reckoning",
                f"torso={torso_name}",
                *ENV_OVERRIDES,
            ])
            horizon = cfg.environment.kwargs.horizon

            if torso_name == TORSOS[0]:
                print("Setting up environment...")
                env, env_params = make_env(**cfg.environment)
                plot_dir = make_plot_dir(cfg)

            label = LABELS[torso_name]
            cell_cls = CELL_CLS[torso_name]
            print(f"\n--- {label} (seed {SEED}) ---")

            net = build_network(cfg, env, env_params)
            params = load_params(cfg, torso_name, seed=SEED)

            # Single-env step function for diagnostic plots
            step_fn = make_step_fn(net, cell_cls)
            init_carry_1 = net.initialize_carry((1, None))

            # Warm up single-env
            _obs, _ = env.reset(jax.random.key(0), env_params)
            step_fn(params, _obs, jnp.array(0, jnp.int32),
                    jnp.array(True, jnp.bool_), jnp.array(0.0), init_carry_1)

            # Diagnostic plot: single episode
            ranks, innovations = [], []

            def on_step(t, carry, state, _ranks=ranks, _innovations=innovations):
                _ranks.append(extract_effective_rank(carry))
                _innovations.append(extract_innovation_norm(state))

            print(f"  Running single episode (diagnostic plots)...")
            goal_dist, reached = run_episode(
                step_fn, params, init_carry_1, env, env_params, NORM_ENV_SEED,
                horizon, on_step=on_step,
            )
            num_steps = len(ranks)
            print(f"    {num_steps} steps, dist={goal_dist}, reached={reached}")
            norm_data[label] = (ranks, innovations, goal_dist)

            # Batched step function for eval
            batch_step_fn = make_batch_step_fn(net)
            init_carry_N = net.initialize_carry((NUM_ENVS, None))

            # Warm up batched
            _obs_b, _ = reset_batch(env, env_params, 0, NUM_ENVS)
            batch_step_fn(params, _obs_b, jnp.zeros(NUM_ENVS, jnp.int32),
                          jnp.ones(NUM_ENVS, jnp.bool_), jnp.zeros(NUM_ENVS), init_carry_N)

            # Eval: 1024 parallel episodes
            print(f"  Evaluating ({NUM_ENVS} episodes)...")
            results = run_eval_episodes(
                batch_step_fn, params, init_carry_N, env, env_params, SEED, horizon,
            )
            success = np.mean([r for _, r in results])
            print(f"    Success: {success:.1%} ({int(success * NUM_ENVS)}/{NUM_ENVS})")
            eval_data[label] = results

    # ── Plots ──
    print("\n=== Generating plots ===")
    salt_ranks, salt_innov, salt_dist = norm_data["SALT"]
    delta_ranks, delta_innov, delta_dist = norm_data["Gated DeltaNet"]
    plot_hidden_state_diagnostics(salt_ranks, salt_innov, delta_ranks, delta_innov, salt_dist, delta_dist, plot_dir)
    plot_performance_profile(eval_data["SALT"], eval_data["Gated DeltaNet"], plot_dir)

    # Summary
    for name in ["SALT", "Gated DeltaNet"]:
        overall = np.mean([r for _, r in eval_data[name]])
        print(f"{name}: {overall:.1%} success ({NUM_ENVS} episodes)")

    print("Done!")


if __name__ == "__main__":
    main()
