#!/bin/bash

uv run main.py --multirun \
    algorithm=pqn \
    environment=grimax/piloting \
    logger=wandb \
    logger.loggers.wandb.entity=noahfarr \
    logger.loggers.wandb.project=salt \
    num_seeds=5 \
    torso=salt,linear_attention \
