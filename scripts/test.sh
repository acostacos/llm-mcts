export PYTHONPATH="$PWD"

# python vh/data_gene/gen_data/vh_init.py --display 0 --port "8083" --task all --mode simple --usage train --num-per-apartment 500 
# python vh/data_gene/testing_agents/gene_data.py --mode simple \
#    --dataset_path ./vh/dataset/env_task_set_500_simple.pik\
#    --base-port 8104 

# python vh/data_gene/gen_data/vh_init.py --port "8083" --task all --mode full --usage train --num-per-apartment 500 
# python vh/data_gene/testing_agents/gene_data.py --mode full \
#    --dataset_path ./vh/dataset/env_task_set_500_full.pik\
#    --base-port 8104 

python mcts/virtualhome/mcts_agent.py \
    --exploration_constant 24 \
    --max_episode_len 50 \
    --max_depth 20 \
    --round 0 \
    --simulation_per_act 2 \
    --simulation_num 100 \
    --discount_factor 0.95  \
    --uct_type PUCT \
    --mode simple \
    --seen_item \
    --seen_apartment\
    --model gpt-3.5-turbo-0125 \
    --seen_comp
