#!/usr/bin/env bash
set -euo pipefail

uv run main.py -m \
  algorithm=dqn \
  environment=minatar/asterix,minatar/breakout,minatar/freeway \
  torso=mlp,gru,linear_attention \
  seed=0 \
  num_seeds=2 \
  'logger=[dashboard,wandb]' \
  logger.loggers.wandb.entity=noahfarr \
  logger.loggers.wandb.project=memorax
