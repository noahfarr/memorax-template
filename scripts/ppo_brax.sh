#!/usr/bin/env bash
set -euo pipefail

uv run main.py -m \
  torso=mlp,gru,gated_linear_attention,gated_delta_net,salt \
  environment=brax/ant,brax/half_cheetah,brax/hopper,brax/walker \
  seed=0 \
  'logger=[dashboard,wandb]' \
  logger.loggers.wandb.entity=noahfarr \
  logger.loggers.wandb.project=salt
