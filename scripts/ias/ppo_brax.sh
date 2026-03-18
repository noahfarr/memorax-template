#!/usr/bin/env bash
set -euo pipefail

uv run main.py --multirun hydra/launcher=ias \
  torso=mlp,gru,delta_net,gated_delta_net,salt \
  environment=brax/ant,brax/half_cheetah,brax/hopper,brax/walker \
  ++environment.kwargs.mode=V \
  seed=1 \
  logger=wandb
