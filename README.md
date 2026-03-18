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

## Configuration

All configuration lives in `config/` and is composed via Hydra defaults:

```
config/
├── config.yaml              # Top-level defaults
├── algorithm/                # PPO, PQN
├── environment/              # Gymnax, MinAtar, Brax, PopJym, PopGym Arcade, ...
├── hyperparameters/          # Per-algorithm, per-environment hyperparameter overrides
├── torso/                    # Sequence model architectures (GRU, MLP, Linear Attention, ...)
└── logger/                   # Logging backends (CLI dashboard, W&B)
```

### Algorithms

| Config | Algorithm |
|--------|-----------|
| `ppo` | [PPO](https://arxiv.org/abs/1707.06347) |
| `pqn` | [PQN](https://arxiv.org/abs/2407.04811) |
| `dqn` | [PQN](https://arxiv.org/abs/2407.04811) |
| `sac` | [PQN](https://arxiv.org/abs/2407.04811) |
| `ac_lambda` | [PQN](https://arxiv.org/abs/2407.04811) |

### Torso Architectures

| Config | Architecture |
|--------|-------------|
| `mlp` | Feedforward (MLP) |
| `gru` | [GRU](https://arxiv.org/abs/1406.1078) |
| `linear_attention` | [Linear Attention](https://arxiv.org/abs/2006.16236) |
| `gated_linear_attention` | [Gated Linear Attention](https://arxiv.org/abs/2312.06635) |
| `delta_net` | [Delta Net](https://arxiv.org/abs/2102.11174) |
| `gated_delta_net` | [Gated Delta Net](https://arxiv.org/abs/2412.06464) |

### Environments

Environments span several JAX-native suites:

- **Gymnax**: CartPole
- **MinAtar**: Asterix, Breakout, Freeway
- **Brax**: Ant, HalfCheetah, Hopper, Walker
- **PopJym**: Autoencode, Battleship, Concentration, Count Recall, Higher Lower, Minesweeper, Repeat First/Previous, Stateless CartPole/Pendulum, ...
- **PopGym Arcade**: Autoencode, Battleship, Breakout, CartPole, Count Recall, Minesweeper, Navigator, Skittles, Tetris
- **Craftax**: Symbolic
- **Navix**: Maze
- **Grimax**: Dead Reckoning, Odometer, Piloting
- **POBAX**: Anna's Maze, Pocman, RockSample, T-Maze, ...
- **BSuite**: Memory Chain

## Project Structure

```
memorax-template/
├── main.py                   # Training entry point
├── config/                   # Hydra configs
├── scripts/                  # Shell scripts for common experiments
└── src/
    ├── algorithms/           # Algorithm-environment wiring
    ├── environment.py        # Environment factory
    └── utils/                # Hydra resolvers
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.
