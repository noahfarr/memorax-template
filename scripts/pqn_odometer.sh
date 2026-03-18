#!/usr/bin/env bash
set -euo pipefail

uv run main.py -m \
  algorithm=pqn \
  environment=grimax/odometer \
  logger=wandb \
  logger.loggers.wandb.entity=noahfarr \
  logger.loggers.wandb.project=dead_reckoning \
  torso=mlp,gru,delta_net,gated_delta_net,salt,salt_diag \
  num_seeds=3
