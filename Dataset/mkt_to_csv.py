from collections import defaultdict
import pickle

data = []

with open("wiki2009.mtx") as f:
    data = list(map(lambda l: l.rstrip().split(), f.readlines()[2:]))
    # for i in f:
    #     data.append(i.rstrip().split())


# print(data[0][1])

def get_node_index(name, edges):
    for i in range(len(edges)): 
        if edges[i][0] == name: return i
    
    return -1

edges_dict = defaultdict(list)

for i, item in enumerate(data): 
    edges_dict[item[0]].append(item[1])
    
    print(f"added element {i} of {len(data)}.")

result = []

for start, end in edges_dict.items():
    result.append(start + ',' + ','.join(end) + '\n')

pickle.dump(edges_dict, open( "save.p", "wb" ))

with open("wiki2009.csv", "w") as of: 
    of.writelines(result)