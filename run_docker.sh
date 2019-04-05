#!/bin/bash
HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
name=${USER}_pymarl_${HASH}

docker run \
    --name $name \
    --user $(id -u):$(id -g) \
    -v `pwd`:/pymarl \
    -t pymarl:1.0 \
    python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3m