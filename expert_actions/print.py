import pickle
import pprint

with open('expert_actions/obs_langs_list.pik', 'rb') as file:
    object = pickle.load(file)
    print(object[0])
    # print(object.keys())

