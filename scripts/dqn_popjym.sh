#!/usr/bin/env bash
set -euo pipefail

uv run main.py -m \
  algorithm=dqn \
  environment=popjym/autoencode_easy,popjym/autoencode_medium,popjym/autoencode_hard,popjym/concentration_easy,popjym/concentration_medium,popjym/concentration_hard,popjym/count_recall_easy,popjym/count_recall_medium,popjym/count_recall_hard,popjym/higher_lower_easy,popjym/higher_lower_medium,popjym/higher_lower_hard,popjym/repeat_first_easy,popjym/repeat_first_medium,popjym/repeat_first_hard,popjym/repeat_previous_easy,popjym/repeat_previous_medium,popjym/repeat_previous_hard \
  torso=mlp,gru,linear_attention \
  seed=0 \
  num_seeds=2 \
  'logger=[dashboard,wandb]' \
  logger.loggers.wandb.entity=noahfarr \
  logger.loggers.wandb.project=memorax
