#!/usr/bin/env bash
set -euo pipefail

uv run main.py -m \
  algorithm=pqn \
  environment=popjym/autoencode_easy,popjym/autoencode_medium,popjym/autoencode_hard,popjym/concentration_easy,popjym/concentration_medium,popjym/concentration_hard,popjym/count_recall_easy,popjym/count_recall_medium,popjym/count_recall_hard,popjym/higher_lower_easy,popjym/higher_lower_medium,popjym/higher_lower_hard,popjym/repeat_first_easy,popjym/repeat_first_medium,popjym/repeat_first_hard,popjym/repeat_previous_easy,popjym/repeat_previous_medium,popjym/repeat_previous_hard,popjym/battleship_easy,popjym/battleship_medium,popjym/battleship_hard,popjym/stateless_cartpole_easy,popjym/stateless_cartpole_medium,popjym/stateless_cartpole_hard,popjym/noisy_stateless_cartpole_easy,popjym/noisy_stateless_cartpole_medium,popjym/noisy_stateless_cartpole_hard,popjym/minesweeper_easy,popjym/minesweeper_medium,popjym/minesweeper_hard,popjym/multiarmed_bandit_easy,popjym/multiarmed_bandit_medium,popjym/multiarmed_bandit_hard \
  torso=mlp,gru,salt,delta_net,gated_delta_net \
  seed=3 \
  num_seeds=2 \
  'logger=[dashboard,wandb]' \
  logger.loggers.wandb.entity=noahfarr \
  logger.loggers.wandb.project=salt
