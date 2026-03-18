#!/usr/bin/env bash
set -euo pipefail

# TORSOS="delta_net gated_delta_net linear_transformer salt"
TORSOS="mlp"

for t in $TORSOS; do
    echo "Running experiment with torso: $t"
    uv run main.py --multirun hydra/launcher=submitit_local \
      torso@actor.torso=$t \
      torso@critic.torso=$t \
      environment="brax/ant","brax/half_cheetah" \
      ++environment.kwargs.mode=P \
      seed=1,2 \
      logger=[dashboard,wandb]
done
