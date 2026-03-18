#!/usr/bin/env bash
set -euo pipefail

uv run main.py -m \
  algorithm=pqn \
  environment=poatar/breakout \
  torso=salt_diag \
  embeddings=oa \
  num_seeds=1 \
  'logger=[dashboard,wandb]'
