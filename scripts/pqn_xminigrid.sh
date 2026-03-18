#!/usr/bin/env bash
set -euo pipefail

uv run main.py -m \
  algorithm=pqn \
  environment=xminigrid/empty_5x5,xminigrid/empty_16x16,xminigrid/memory_9x9,xminigrid/four_rooms,xminigrid/doorkey_5x5,xminigrid/locked_room \
  torso=mlp,gru,linear_attention \
  seed=0 \
  num_seeds=2 \
  'logger=[dashboard,wandb]' \
  logger.loggers.wandb.entity=noahfarr \
  logger.loggers.wandb.project=memorax
