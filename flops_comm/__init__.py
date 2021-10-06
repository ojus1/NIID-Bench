from .flops import compute_flops
from .comm_cost import compute_comm_cost

def flatten(d):
    new = {}
    for k in d[0].keys():
        new[k] = []
    
    for item in d:
        for k in new.keys():
            new[k].append(item[k])
    return new