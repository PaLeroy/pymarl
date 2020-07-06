#!/usr/bin/env bash
source env/bin/activate
set -x
file1="/home/pascal/sc2/backup_results_wartank/models/population_alone_maven_$1/agent_id_0"
file2="/home/pascal/sc2/backup_results_wartank/models/population_maven_vs_heuristic_$1/agent_id_0"
name_var2="population_test_vs_heuristic_maven_vs_heuristic_$1"
name_var1="population_test_vs_heuristic_alone_maven_$1"

python3 src/main_popu_to_test.py \
--config=popu_maven_vs_heuristic_test --env-config=sc2_multi \
with 'env_args.map_name=3m_multi' \
'name='$name_var1 \
'agent_type_1.checkpoint_path=["'$file1'"]'

python3 src/main_popu_to_test.py \
--config=popu_maven_vs_heuristic_test --env-config=sc2_multi \
with 'env_args.map_name=3m_multi' \
'name='$name_var2 \
'agent_type_1.checkpoint_path=["'$file2'"]'

deactivate
