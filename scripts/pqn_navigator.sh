#!/bin/bash

uv run main.py --multirun \
    algorithm=pqn \
    environment=popgym_arcade/navigator/medium \
    ++environment.kwargs.partial_obs=True \
    logger=wandb \
    logger.loggers.wandb.entity=noahfarr \
    logger.loggers.wandb.project=salt \
    num_seeds=1 \
    torso=mlp,delta_net,gated_delta_net,salt,linear_attention \
