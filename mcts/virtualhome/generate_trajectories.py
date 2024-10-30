from mcts.mcts.mcts import MCTSAgent
from mcts.virtualhome.llm_policy import LLMPolicy
from mcts.virtualhome.belief import Belief, container_classes, surface_classes
from mcts.virtualhome.llm_model import LLM_Model
from vh.data_gene.envs.unity_environment import UnityEnvironment
from vh.vh_mdp.vh_graph.envs.vh_env import VhGraphEnv
from vh.learned_policy.utils_bc.utils_interactive_eval import get_valid_actions
from vh.learned_policy.data_loader import get_goal_language
import pickle
import argparse
import time
import copy
import random
import yaml
import json

def get_valid_action_alt(obs, agent_id=0):
    valid_action_space = []
    valid_action_space_dict = get_valid_actions(obs, agent_id)
    for action in valid_action_space_dict:
        interact_item_idxs = valid_action_space_dict[action]
        action = action.replace('walktowards', 'walk')
        if 'put' in action:
            
            valid_action_space += [
                f'[{action}] <{grab_name}> ({grab_id}) <{item_name}> ({item_id})'
                    for grab_id, grab_name, item_id, item_name in interact_item_idxs]
        else:
            valid_action_space += [
                f'[{action}] <{item_name}> ({item_id})'
                    for item_id, item_name in interact_item_idxs if item_name not in ['wall', 'floor', 'ceiling', 'curtain', 'window']]
            
    return valid_action_space

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--num_test', type=int, default=None)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--seen_item', type=bool, default=False)
    parser.add_argument('--seen_apartment', type=bool, default=False)
    parser.add_argument('--seen_comp', type=bool, default=False)
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125')
    
    return parser.parse_args()

def find_test_data_file_path(args):
    
    if args.load_path:
        file_path = args.load_path
        
    elif args.mode and args.num_test:
        file_path = f'./vh/dataset/env_task_set_{args.num_test}_{args.mode}_'
        
        if not args.seen_item:
            file_path += 'unseen_item.pik'
        elif not args.seen_apartment:
            file_path += 'unseen_apartment.pik'
        elif not args.seen_comp:
            file_path += 'unseen_composition.pik'
        else:
            file_path += 'seen.pik'
        
    else:
        raise ValueError('No mode or load_path specified')
        
    return file_path

def test():
    args = parse_args()
    file_path = find_test_data_file_path(args)
    env_task_set = pickle.load(open(file_path, 'rb'))
    executable_args = {
                    'file_name': "./vh/vh_sim/simulation/unity_simulator/v2.2.5/linux_exec.v2.2.5_beta.x86_64",
                    'x_display': "0",
                    'no_graphics': True
    }
    llm_model = LLM_Model("cuda:0", args.model)
    vhenv = UnityEnvironment(num_agents=1,
                                max_episode_length=100,
                                port_id=2,
                                env_task_set=env_task_set,
                                observation_types=["partial"],
                                use_editor=False,
                                executable_args=executable_args,
                                base_port=8084)
    goal_spec = vhenv.get_goal(vhenv.task_goal[0], vhenv.agent_goals[0])
    graph = vhenv.get_graph()
    container_name2id = {}
    
    tasks_infos = []

    for node in graph['nodes']:
        if node['class_name'] in container_classes or node['class_name'] in surface_classes:
            container_name2id[node['class_name']] = node['id']
    
    history = []
    done = False
    succ = 0
    total = 0
    for i in range(len(vhenv.env_task_set)):

        obs = vhenv.reset(task_id=i)
        goal_spec = vhenv.get_goal(vhenv.task_goal[0], vhenv.agent_goals[0])
        
        task_info = {
            'goal': get_goal_language(vhenv.task_goal[0], graph),
            'goal_spec': goal_spec,
            'task_id': i,
            'states': [obs[0]],
            'actions': []
        }

        graph = vhenv.get_graph() 
        history = []
        
        valid_actions = get_valid_action_alt(obs, 0)

        done = False
        for i in range(30):
            print(" ---------------------- Step: ", i, " ---------------------- ")
            
            action = random.choice(valid_actions)
            
            # action = agent.search(obs, history, i, valid_actions, done)
            # action = agent.llm_policy.act(history, obs, valid_actions, agent.env.get_goal()) 
            
            # graph = vhenv.get_graph()
            
            print("Action: ", action)

            obs, reward, done, info, success = vhenv.step({0: action})
            
            task_info['states'].append(obs[0])
            task_info['actions'].append(action)
            
            # agent.env.update_(action, obs[0]) 
            # valid_actions = agent.env.get_valid_action(obs)
            valid_actions = get_valid_action_alt(obs, 0)
            history.append(action)
            
            if done:
                succ += 1
                break
        total += 1
        task_info['success'] = success
        tasks_infos.append(task_info)
        time.sleep(5)
        print("succ rate: ", succ / total)
        
    pickle.dump(tasks_infos, open('task_trajectories/1_simple_seen.pik', 'wb'))
    #save as yaml
    with open('task_trajectories/1_simple_seen.yaml', 'w') as file:
        yaml.dump(tasks_infos, file)
    #save as json
    with open('task_trajectories/1_simple_seen.json', 'w') as file:
        json.dump(tasks_infos, file)

if __name__ == "__main__" :
    test()