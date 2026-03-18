#!/usr/bin/env bash
set -euo pipefail

uv run main.py -m \
  torso@actor.torso=mlp \
  torso@critic.torso=mlp \
  environment=gymnax/cartpole \
  seed=0 \
  num_seeds=1 \
  'logger=[wandb,dashboard]'
