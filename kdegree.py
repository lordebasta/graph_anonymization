from random import randint
import numpy as np
import collections
import networkx as nx
import sys
import os
import matplotlib


def compute_I(d):
    d_i = d[0]
    i_value = 0
    for d_j in d:
        i_value += (d_i - d_j)
    return i_value

def compute_da(d, k, t_list=[]):
    # costs = [[compute_da(d[:t], k)[0]+compute_I(d[t:]), t] for t in range(k, len(d)-k+1)]
    # costs.append([compute_I(d), 0])
    # print(costs)
    # return min(costs, key=lambda x: x[0])
    # print(type(t_list))
    if len(d) < 2*k: 
        t_list.insert(0, 0)
        return (compute_I(d), t_list)
    costs = []
    for t in range(k, len(d)-k+1):
        costs.append( [compute_da(d[:t], k, t_list)[0]+compute_I(d[t:]), t])
    costs.append([compute_I(d), 0])
    # print(costs)
    minimum = min(costs, key=lambda x: x[0])
    t_list.insert(0, minimum[1])
    return (minimum[0], t_list)

def c_merge(d, d1, k):
    c_merge_cost = d1 - d[k] + compute_I(d[k+1:min(len(d), 2*k)])
    return c_merge_cost


def c_new(d, k):
    t = d[k:min(len(d), 2*k-1)]
    c_new_cost = compute_I(t)
    return c_new_cost


def greedy_rec_algorithm(d, array_degrees, k_degree, pos_init, extension):
    if pos_init + extension >= len(array_degrees) - 1:
        for i in range(pos_init, len(array_degrees)):
            array_degrees[i] = array_degrees[pos_init]
        return array_degrees

    else:
        d1 = array_degrees[pos_init]
        c_merge_cost = c_merge(array_degrees, d1, pos_init + extension)
        c_new_cost = c_new(d, pos_init + extension)
        if c_merge_cost > c_new_cost:
            for i in range(pos_init, pos_init + extension):
                array_degrees[i] = d1
            greedy_rec_algorithm(d, array_degrees, k_degree, pos_init + extension, k_degree)
        else:
            greedy_rec_algorithm(d, array_degrees, k_degree, pos_init, extension + 1)



def dp_graph_anonymization(k, d):
    if len(d) < 2*k: return [d[0] for i in d]
    
    return

def apply_I(d): 
    result = [d[0] for i in d]
    return result

def apply_da(d, k, t_list): 
    for t in t_list: 
        if t == 0: return apply_I(d)
        else:
            result= apply_da(d[:t], k, t_list[1:])
            result.extend(apply_I(d[t:]))
            return result


def construct_graph(tab_index, anonymized_degree):
    graph = nx.Graph()
    if sum(anonymized_degree) % 2 == 1:
        return None

    while True:
        if not all(di >= 0 for di in anonymized_degree):
            return None
        if all(di == 0 for di in anonymized_degree):
            return graph
        v = np.random.choice((np.where(np.array(anonymized_degree) > 0))[0])
        dv = anonymized_degree[v]
        anonymized_degree[v] = 0
        for index in np.argsort(anonymized_degree)[-dv:][::-1]:
            if index == v:
                return None
            if not graph.has_edge(tab_index[v], tab_index[index]):
                graph.add_edge(tab_index[v], tab_index[index])
                anonymized_degree[index] = anonymized_degree[index] - 1


if __name__ == "__main__":
    d = [3, 3, 3, 2, 1]
    t_list = compute_da(d, 2)[1]
    d1 = apply_da(d, 2, t_list)
    print(d1)
