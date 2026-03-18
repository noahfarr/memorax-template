#!/usr/bin/env bash
set -euo pipefail

uv run main.py -m \
  algorithm=pqn \
  environment=minatar/breakout \
  torso=delta_net,gated_delta_net,salt,salt_diag,salt_lr,salt_f \
  embeddings=oa \
  num_seeds=3 \
  'logger=[dashboard,wandb]'
