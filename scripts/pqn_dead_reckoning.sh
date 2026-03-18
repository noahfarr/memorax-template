#!/bin/bash

uv run main.py --multirun \
    algorithm=pqn \
    environment=grimax/dead_reckoning \
    logger=wandb \
    logger.loggers.wandb.entity=noahfarr \
    logger.loggers.wandb.project=salt \
    num_seeds=5 \
    torso=gated_linear_attention \
