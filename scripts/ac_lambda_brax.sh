#!/usr/bin/env bash
set -euo pipefail

uv run main.py -m \
  algorithm=ac_lambda \
  environment=brax/ant,brax/half_cheetah,brax/hopper,brax/walker \
  torso=mlp,gru,linear_attention \
  seed=0 \
  'logger=[dashboard,wandb]' \
  logger.loggers.wandb.entity=noahfarr \
  logger.loggers.wandb.project=memorax
