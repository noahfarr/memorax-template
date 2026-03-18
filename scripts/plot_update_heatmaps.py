"""Visualize hidden state update matrices (ΔS) at key timesteps.

Loads trained GRU, Gated DeltaNet, and SALT checkpoints, runs a single
evaluation episode on Dead Reckoning, and produces a 3×5 heatmap grid:

  Rows:    GRU (top), Gated DeltaNet (middle), SALT (bottom)
  Columns: t=1, t=5, t=10, t=25, t=50

Each panel shows |ΔS_t| = |S_t - S_{t-1}|.
- For Memoroid models (SALT, GDN): layer 0, head 0 → 16×16 matrix.
- For GRU: full hidden state (512-dim) reshaped to 16×32.
"""

import os
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.environ["PYTHONUNBUFFERED"] = "1"

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
import seaborn as sns
from fanda.fanda import Fanda
from fanda.utils import close_fig, save_fig
from fanda.visualizations import annotate_axis, decorate_axis
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from matplotlib.colors import SymLogNorm
from memorax.networks import Network, heads
from memorax.networks.blocks import FFN, GatedResidual, PreNorm, Stack

import salt.utils.resolvers  # noqa: F401 — registers OmegaConf resolvers
from salt.environments.environment import make as make_env
from salt.networks import SelectiveFeatureExtractor

# ── Constants ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = str((ROOT / "config").resolve())

TORSOS = ["gru", "gated_delta_net", "salt"]
LABELS = {"gru": "GRU", "gated_delta_net": "Gated DeltaNet", "salt": "SALT"}

ENV_OVERRIDES = [
    "environment.kwargs.grid_size=64",
    "environment.kwargs.min_goal_distance=32",
    "environment.kwargs.horizon=200",
]

SEED = 42  # single-episode seed

TIMESTEPS = (1, 5, 10, 25, 50)  # 1-indexed timesteps to visualize

flatten = lambda x: x.reshape(*x.shape[:2], -1).astype(jnp.float32)

EXTRACT_STATE = {
    "salt": lambda carry: np.array(carry[0][1][0, 0, 0]),
    "gated_delta_net": lambda carry: np.array(carry[0][1][0, 0, 0]),
    "gru": lambda carry: np.array(carry[0][0]).reshape(16, 32),
}


def make_plot_dir(cfg):
    """Derive plot output directory from environment config."""
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', cfg.environment.env_id.split("-")[0]).lower()
    plot_dir = ROOT / "plots" / cfg.environment.namespace / name
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


# ── Checkpoint loading ─────────────────────────────────────────────────────


def load_params(cfg, torso_name, seed=0):
    ckpt_root = ROOT / "checkpoints" / cfg.environment.namespace / cfg.environment.env_id / "pqn" / torso_name / str(seed)
    ckpt_path = sorted(ckpt_root.iterdir())[-1]
    print(f"  Loading {torso_name} seed {seed} from {ckpt_path}")
    return ocp.StandardCheckpointer().restore(str(ckpt_path))["params"]


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


# ── Forward pass ───────────────────────────────────────────────────────────


def make_step_fn(q_network):
    @jax.jit
    def step_fn(params, obs, prev_action, prev_done, prev_reward, carry):
        obs_seq = obs[None, None]
        action_seq = prev_action[None, None]
        done_seq = prev_done[None, None]
        reward_seq = prev_reward[None, None, None]
        (new_carry, (q_values, _)), _ = q_network.apply(
            params,
            observation=obs_seq,
            mask=done_seq,
            action=action_seq,
            reward=reward_seq,
            done=done_seq,
            initial_carry=carry,
            mutable=["intermediates"],
        )
        return new_carry, q_values[0, 0]
    return step_fn


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


# ── Plot ───────────────────────────────────────────────────────────────────


def plot_update_heatmaps(all_updates, row_labels, plot_dir, timesteps=TIMESTEPS):
    """Create an NxM heatmap grid of |ΔS_t|.

    Args:
        all_updates: list of update lists, one per row (model)
        row_labels: list of model names
        timesteps: which timesteps to show as columns

    Single shared symlog colour scale so both the large t=1 updates and
    small blind-phase updates are visible.
    """
    nrows = len(row_labels)

    # Gather |ΔS| matrices at the requested timesteps (1-indexed → 0-indexed)
    mats = [[np.abs(row[t - 1]) for t in timesteps] for row in all_updates]

    # Single shared vmax across all panels
    vmax = max(m.max() for row in mats for m in row)

    # SymLogNorm: linear below linthresh, logarithmic above.
    norm = SymLogNorm(linthresh=0.01, vmin=0, vmax=vmax)

    ncols = len(timesteps)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.6 * ncols, 2.3 * nrows + 0.8),
        gridspec_kw={"wspace": 0.05, "hspace": 0.15},
    )

    im = None
    for col_idx, t in enumerate(timesteps):
        for row_idx in range(nrows):
            ax = axes[row_idx, col_idx]
            sns.heatmap(
                mats[row_idx][col_idx],
                ax=ax,
                cmap="inferno",
                norm=norm,
                cbar=False,
                xticklabels=False,
                yticklabels=False,
            )
            im = ax.collections[0]
            title = f"$t = {t}$" if row_idx == 0 else ""
            ylabel = row_labels[row_idx] if col_idx == 0 else ""
            (
                Fanda(fig=fig, ax=ax)
                .pipe(annotate_axis, xlabel="", ylabel=ylabel, title=title)
                .pipe(decorate_axis, spines=["bottom", "left", "top", "right"],
                      ticklabelsize="small")
            )

    # Single colorbar spanning entire figure
    cb = fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        location="bottom",
        fraction=0.04,
        pad=0.06,
        aspect=40,
        label=r"$|\Delta \mathbf{S}_t|$",
    )
    cb.ax.tick_params(labelsize=9)

    (
        Fanda(fig=fig, ax=axes[0, 0])
        .pipe(save_fig, name=str(plot_dir / "update_heatmaps"))
        .pipe(close_fig)
    )


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        all_updates = []
        row_labels = []
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
            print(f"\n--- {label} ---")

            net = build_network(cfg, env, env_params)
            params = load_params(cfg, torso_name, seed=0)

            step_fn = make_step_fn(net)
            init_carry = net.initialize_carry((1, None))

            # Warm up JIT
            _obs, _ = env.reset(jax.random.key(0), env_params)
            step_fn(params, _obs, jnp.array(0, jnp.int32),
                    jnp.array(True, jnp.bool_), jnp.array(0.0), init_carry)

            extract_fn = EXTRACT_STATE[torso_name]
            prev_S = [extract_fn(init_carry)]
            updates = []

            def on_step(t, carry, *_, _extract=extract_fn, _prev=prev_S, _upd=updates):
                S_t = _extract(carry)
                _upd.append(S_t - _prev[0])
                _prev[0] = S_t

            print(f"  Running single episode (seed={SEED})...")
            goal_dist, reached = run_episode(
                step_fn, params, init_carry, env, env_params, SEED, horizon,
                on_step=on_step,
            )
            print(f"    {len(updates)} steps, dist={goal_dist}, reached={reached}")

            for t in TIMESTEPS:
                if t <= len(updates):
                    mag = np.abs(updates[t - 1]).max()
                    print(f"    t={t}: max|ΔS| = {mag:.6f}")

            all_updates.append(updates)
            row_labels.append(label)

    print("\n=== Generating heatmap figure ===")
    plot_update_heatmaps(all_updates, row_labels, plot_dir)
    print("Done!")


if __name__ == "__main__":
    main()
