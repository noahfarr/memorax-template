#!/usr/bin/env bash
set -euo pipefail

uv run main.py -m \
  algorithm=pqn \
  environment=navix/maze_01,navix/maze_02,navix/maze_03 \
  torso=mlp,salt,delta_net,gated_delta_net,gru \
  num_seeds=3 \
  'logger=[dashboard,wandb]' \
  logger.loggers.wandb.entity=noahfarr \
  logger.loggers.wandb.project=salt
