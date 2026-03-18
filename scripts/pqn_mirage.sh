#!/usr/bin/env bash
set -euo pipefail

uv run main.py -m \
  algorithm=pqn \
  environment=mirage/dmts,mirage/n_back,mirage/paired_associate,mirage/change_detection,mirage/transitive_inference,mirage/context_dependent,mirage/wm_updating,mirage/complex_span \
  torso=mlp,gru,salt,delta_net,gated_delta_net,linear_attention \
  num_seeds=1 \
  'logger=[dashboard,wandb]' \
  logger.loggers.wandb.entity=noahfarr \
  logger.loggers.wandb.project=salt \
