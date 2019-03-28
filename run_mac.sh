#!/usr/bin/env bash
source env/bin/activate
python3 src/main.py --config=qmix_smac --env-config=sc2multi with env_args.map_name=3m_multi_close
deactivate