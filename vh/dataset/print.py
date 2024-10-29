import pickle
import pprint

with open('env_task_set_500_simple.pik', 'rb') as file:
    object = pickle.load(file)
    print(len(object))
    first_object = object[0]
    print(type(first_object))
    print(first_object.keys())
