#!/usr/bin/env bash
source env/bin/activate
set -x
file1="/home/pascal/sc2/results/models/population_alone_$3_$1/agent_id_0"
file2="/home/pascal/sc2/results/models/population_$3_vs_heuristic_$2/agent_id_0"
name_var="population_test_duo_$3_A$1_vs_H$2"
timeout 15m python3 src/main_popu_to_test.py \
--config=popu_duo_$3_test --env-config=sc2_multi \
with 'env_args.map_name=3m_multi' \
'name='$name_var \
'agent_type_1.checkpoint_path=["'$file1'", "'$file2'"]'
deactivate