#!/usr/bin/env bash
set -euo pipefail

uv run main.py -m \
  algorithm=pqn \
  environment=popgym_arcade/count_recall/easy,popgym_arcade/battleship/easy,popgym_arcade/minesweeper/easy,popgym_arcade/navigator/easy,popgym_arcade/autoencode/easy \
  torso=mlp,gru,gated_delta_net,salt \
  num_seeds=1 \
  'logger=[dashboard,wandb]'
