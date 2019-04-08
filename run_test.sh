#!/usr/bin/env bash
source env/bin/activate
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3m use_tensorboard=True
deactivate
