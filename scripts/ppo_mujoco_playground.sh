#!/usr/bin/env bash
set -euo pipefail

uv run main.py -m \
  algorithm=ppo \
  environment=mujoco_playground/ant,mujoco_playground/humanoid,mujoco_playground/cartpole_balance,mujoco_playground/cartpole_swingup \
  torso=mlp,gru,linear_attention \
  seed=0 \
  'logger=[dashboard,wandb]' \
  logger.loggers.wandb.entity=noahfarr \
  logger.loggers.wandb.project=memorax
