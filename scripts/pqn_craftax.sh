#!/usr/bin/env bash
set -euo pipefail

uv run main.py -m \
  algorithm=pqn \
  environment=craftax/symbolic \
  torso=mlp,gru,salt,delta_net,gated_delta_net \
  num_seeds=1 \
  'logger=[dashboard,wandb]' \
  logger.loggers.wandb.entity=noahfarr \
  logger.loggers.wandb.project=salt
