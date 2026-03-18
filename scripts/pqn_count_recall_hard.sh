#!/usr/bin/env bash
set -euo pipefail

uv run main.py -m \
  algorithm=pqn \
  environment=popjym/count_recall_hard \
  hyperparameters=pqn/popjym \
  logger=wandb \
  logger.loggers.wandb.entity=noahfarr \
  logger.loggers.wandb.project=count_recall_hard \
  torso=mlp,gru,delta_net,gated_delta_net,salt,salt_diag \
  num_seeds=3
