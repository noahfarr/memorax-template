# memorax-template

A template project for running memory-augmented reinforcement learning experiments with [Memorax](https://github.com/memory-rl/memorax). It provides a [Hydra](https://hydra.cc/)-based configuration system for mixing and matching algorithms, environments, torso architectures, and hyperparameters.

## Installation

```bash
uv sync
```

For CUDA support:

```bash
uv sync --extra cuda
```

Optionally, set up Weights & Biases for logging:

```bash
wandb login
```

## Usage

Train an agent using Hydra config overrides:

```bash
python main.py algorithm=pqn environment=minatar/asterix torso=gru
```

Override any parameter from the command line:

```bash
python main.py algorithm=ppo environment=brax/ant torso=mlp total_timesteps=5_000_000 num_seeds=3
```

### Renaming the Template

To rename the project for your own use:

```bash
python rename_template.py my-project-name
```

## Configuration

All configuration lives in `config/` and is composed via Hydra defaults:

```
config/
├── config.yaml              # Top-level defaults
├── algorithm/                # PPO, PQN, DQN, SAC, AC(λ), Gradient PPO, MAPPO, R2D2
├── environment/              # Gymnax, MinAtar, Brax, PopJym, PopGym Arcade, ...
├── hyperparameters/          # Per-algorithm, per-environment hyperparameter overrides
├── torso/                    # Sequence model architectures (MLP, GRU, Mamba, S5, ...)
└── logger/                   # Logging backends (CLI dashboard, W&B)
```

### Algorithms

| Config | Algorithm |
|--------|-----------|
| `ppo` | [PPO](https://arxiv.org/abs/1707.06347) |
| `pqn` | [PQN](https://arxiv.org/abs/2407.04811) |
| `dqn` | [DQN](https://arxiv.org/abs/1312.5602) |
| `sac` | [SAC](https://arxiv.org/abs/1801.01290) |
| `ac_lambda` | [AC(λ)](https://arxiv.org/abs/2410.14606) |
| `gradient_ppo` | [Gradient PPO](https://arxiv.org/abs/2507.09087) |
| `mappo` | [MAPPO](https://arxiv.org/abs/2103.01955) |
| `r2d2` | [R2D2](https://openreview.net/forum?id=r1lyTjAqYX) |

### Torso Architectures

| Config | Architecture |
|--------|-------------|
| `mlp` | Feedforward (MLP) |
| `gru` | [GRU](https://arxiv.org/abs/1406.1078) |
| `min_gru` | [minGRU](https://arxiv.org/abs/2410.01201) |
| `linear_attention` | [Linear Attention](https://arxiv.org/abs/2006.16236) |
| `self_attention` | [Self-Attention](https://arxiv.org/abs/1706.03762) |
| `lru` | [LRU](https://arxiv.org/abs/2303.06349) |
| `s5` | [S5](https://arxiv.org/abs/2208.04933) |
| `mamba` | [Mamba](https://arxiv.org/abs/2312.00752) |
| `ffm` | [FFM](https://arxiv.org/abs/2310.04128) |
| `rtu` | [RTU](https://arxiv.org/abs/2407.14953) |
| `rtrl` | [RTRL](https://doi.org/10.1162/neco.1989.1.2.270) |
| `shm` | [SHM](https://arxiv.org/abs/2410.10132) |
| `slstm` | [sLSTM](https://arxiv.org/abs/2405.04517) |
| `mlstm` | [mLSTM](https://arxiv.org/abs/2405.04517) |

### Environments

Environments span several JAX-native suites:

- **Gymnax**: CartPole, BSuite (Memory Chain), MinAtar (Asterix, Breakout, Freeway)
- **Gymnasium**: CartPole
- **Brax**: Ant, HalfCheetah, Hopper, Walker
- **MuJoCo Playground**: Ant, CartPole Balance, CartPole Swingup, Humanoid
- **MuJoCo** (via Memorax): Ant
- **ALE**: Breakout
- **GXM Atari**: Breakout
- **JaxMARL**: SMAX
- **PopJym**: Autoencode, Battleship, Concentration, Count Recall, Higher Lower, Minesweeper, Multiarmed Bandit, Noisy Stateless CartPole/Pendulum, Repeat First/Previous, Stateless CartPole/Pendulum (each with Easy/Medium/Hard)
- **PopGym Arcade**: Autoencode, Battleship, Breakout, CartPole, Count Recall, Minesweeper, Navigator, Noisy CartPole, Skittles, Tetris (each with Easy/Medium/Hard)
- **POBAX**: Anna's Maze, Battleship, Pocman, RockSample, Simple Chain, T-Maze, and MuJoCo position/velocity variants (Ant, CartPole, HalfCheetah, Hopper, Pendulum, Walker)
- **XMiniGrid**: DoorKey, Empty (5x5, 16x16), Four Rooms, Locked Room, Memory
- **Craftax**: Symbolic
- **Navix**: Maze (3 layouts)
- **Grimax**: Dead Reckoning, Odometer, Piloting

## Hyperparameter Tuning

The template includes two Hydra sweeper plugins for hyperparameter optimization:

### Optuna

Uses [Optuna](https://optuna.org/) with TPE sampling for Bayesian hyperparameter search:

```bash
python main.py --multirun hydra/sweeper=optuna \
  algorithm=pqn environment=minatar/asterix torso=gru \
  learning_rate="interval(1e-5, 1e-3)" gamma="interval(0.9, 0.999)"
```

### Evosax

Uses [Evosax](https://github.com/RobertTLange/evosax) evolutionary strategies (CMA-ES by default) for population-based search:

```bash
python main.py --multirun hydra/sweeper=evosax \
  algorithm=pqn environment=minatar/asterix torso=gru \
  learning_rate="interval(1e-5, 1e-3)" gamma="interval(0.9, 0.999)"
```

The sweeper configs live in `config/hydra/sweeper/` and can be customized (e.g. number of trials, population size, search strategy).

## Scripts

Pre-configured shell scripts for common experiment sweeps are in `scripts/`:

| Script | Description |
|--------|-------------|
| `ppo_gymnax.sh` | PPO on Gymnax environments |
| `ppo_brax.sh` | PPO on Brax environments |
| `ppo_mujoco_playground.sh` | PPO on MuJoCo Playground environments |
| `pqn_minatar.sh` | PQN on MinAtar games |
| `pqn_popjym.sh` | PQN on PopJym environments |
| `pqn_popgym_arcade.sh` | PQN on PopGym Arcade environments |
| `pqn_craftax.sh` | PQN on Craftax |
| `pqn_xminigrid.sh` | PQN on XMiniGrid environments |
| `dqn_minatar.sh` | DQN on MinAtar games |
| `dqn_popjym.sh` | DQN on PopJym environments |
| `sac_brax.sh` | SAC on Brax environments |
| `sac_mujoco_playground.sh` | SAC on MuJoCo Playground environments |
| `ac_lambda_brax.sh` | AC(λ) on Brax environments |
| `ac_lambda_popjym.sh` | AC(λ) on PopJym environments |

## Plotting

Plot scripts in `scripts/` fetch results from Weights & Biases and generate publication-ready figures using [fanda](https://github.com/noahfarr/fanda). Plots are configured via Hydra configs in `config/plot/`.

```bash
python scripts/plot_minatar.py
```

This generates three types of plots:
- **Interval estimates** — IQM of final returns across runs
- **Sample efficiency curves** — IQM returns over training frames (aggregate and per-environment)
- **Performance profiles** — fraction of runs exceeding a given score threshold

Customize which algorithms, torsos, and environments to include by editing the plot config or via command-line overrides:

```bash
python scripts/plot_minatar.py torsos='[GRU, MLP]' algorithms='[pqn]'
```

Plots are saved to `plots/<environment>/<group_by>/`.

## Project Structure

```
memorax-template/
├── main.py                   # Training entry point
├── rename_template.py        # Rename the template for your project
├── config/                   # Hydra configs
├── scripts/                  # Shell scripts for common experiments
└── src/
    ├── algorithms/           # Algorithm-environment wiring
    ├── environment.py        # Environment factory
    └── utils/                # Hydra resolvers
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.
