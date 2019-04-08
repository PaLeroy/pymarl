#!/usr/bin/env bash
source env/bin/activate
python3 src/main.py --config=iql_smac_multiple --env-config=sc2_multi with env_args.map_name=3m_multi use_tensorboard=True
deactivate
