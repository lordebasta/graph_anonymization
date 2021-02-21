from random import randint
import numpy as np
import collections
import networkx as nx
import sys
import os
import matplotlib
from dataclasses import dataclass


def greedy_algorithm(d, k):
    result = [v for v in d]
    i = 0

    def compute_fill_cost(d, start, k):
        if start + k > len(d):
            return float('inf')
    
        fill_cost = 0
        for i in range(start, start + k):
            fill_cost += d[start] - d[i]
        return fill_cost

    def fill(arr, val, start, end):
        for i in range(start, min(end, len(arr))): arr[i] = val
    
    def compute_merge_cost(d, start, k):
        merge_element_cost =  d[start - 1] - d[start]
        fill_next_cost = compute_fill_cost(d, start + 1, k)
        return merge_element_cost + fill_next_cost
        
    while i < len(d):

        if i+k > len(d): 
            fill(result, result[i-1], i, i+k)
            i += k
        elif i > 0: 
            fill_cost = compute_fill_cost(result, i, k)
            merge_cost = compute_merge_cost(result, i, k)
            if fill_cost >= merge_cost:
                result[i] = result[i-1]
                i += 1
            else: 
                fill(result, result[i], i, i+k)
                i += k
        else: 
            fill(result, result[i], i, i+k)
            i += k  
    
    return result

@dataclass
class Da:
    cost: float
    t: int

def dp_slow_algorithm(d, k): 
    if k == 1: raise ValueError("k == 1 means no anonymization!")

    

    def compute_all_I(d, k):  
        costs_I = [[float('inf') for _ in range(len(d))] for _ in range(len(d) - 1)]
        for i in range(len(d) - 1):
            acc = 0 
            for j in range(i + 1, len(d)):
                acc += d[i] - d[j]
                costs_I[i][j] = acc  
        # for c in costs_I:
        #     print(c)
        return costs_I

    costs_I = compute_all_I(d, k)

    da_list = [Da(cost=float('inf'), t=-1) for _ in d]

    for i in range(len(d)):
        if i < 2*k - 1:
            da_list[i] = Da(cost=costs_I[0][i], t=0)
        else:
            t, cost = min(((t, da_list[t].cost + costs_I[t+1][i]) for t in range(k - 1, i - k + 1)), key= lambda x:x[1])
            if costs_I[0][i]  < cost:
                t = 0
                cost = costs_I[0][i]

            da_list[i] = Da(cost=cost, t=t)
    # def compute_da(d, start, end, k, costs_I): 
    #     length = end - start + 1 

    #     if length < 2*k:
    #         cost = costs_I[start][end]
    #         return cost, [0]
        
    #     costs = []
    #     for t in range (k - 1, end - k + 1):
    #         cost_da, t_list = compute_da(d, 0, t, k, costs_I) 
    #         cost_fill_next = costs_I[t + 1][end]
    #         t_list.insert(0, t)
    #         costs.append((cost_da + cost_fill_next, t_list))
    #     costs.append((costs_I[start][end], [0]))
    #     min_cost, t_list = min(costs, key=lambda c:c[0])
    #     return min_cost, t_list
    
    
    # cost, t_list = compute_da(d, 0, len(d) - 1, k, costs_I)

    t_list = []
    t = da_list[-1].t
    while t > 0:
        t_list.append(t)
        t = da_list[t].t
    t_list.append(0)

    print(t_list)

    result = [v for v in d]
    end = len(result) - 1 
    for t in t_list: 
        if (t == 0): 
            for i in range(t, end + 1):
                result[i] = result[t]
        for i in range(t + 2, end + 1):
            result[i] = result[t + 1]
        end = t

    return result

def dp_algorithm(d, k): 
    if k == 1: raise ValueError("k == 1 means no anonymization!")
    
    def compute_all_I(d, k):  
        costs_I = [[0 for _ in range(min(2*k + i - 1, len(d) - 1))] for i in range(len(d) - 1)]
        for i in range(len(d) - 1):
            acc = 0 
            for j in range(k + i - 1, min(2*k + i - 1, len(d) - 1)):
                acc += d[i] - d[j]
                costs_I[i][j] = acc  
        # for c in costs_I:
        #     print(c)
        return costs_I

    def compute_da(d, start, end, k, costs_I): 
        length = end - start + 1 

        if length < 2*k:
            cost = costs_I[start][end]
            return cost, [0]
        
        costs = []
        for t in range (max(k - 1, end - 2*k + 2), end - k + 1):
            cost_da, t_list = compute_da(d, 0, t, k, costs_I) 
            cost_fill_next = costs_I[t + 1][end]
            t_list.insert(0, t)
            costs.append((cost_da + cost_fill_next, t_list))
        costs.append((costs_I[start][end], [0]))
        min_cost, t_list = min(costs, key=lambda c:c[0])
        return min_cost, t_list

    costs_I = compute_all_I(d, k)
    cost, t_list = compute_da(d, 0, len(d) - 1, k, costs_I)

    result = [v for v in d]
    end = len(result) - 1 
    for t in t_list: 
        if (t == 0): 
            for i in range(t, end + 1):
                result[i] = result[t]
        for i in range(t + 2, end + 1):
            result[i] = result[t + 1]
        end = t

    return result


def compute_I(d):
    d_i = d[0]
    i_value = 0
    for d_j in d:
        i_value += (d_i - d_j)
    return i_value

def compute_da(d, k):
    costs_I = [[0 for _ in range(len(d))] for _ in range(len(d) - 1)]

    for i in range(len(d) - 1):
        acc = 0 
        for j in range(i + 1, len(d)):
            acc += d[i] - d[j]
            costs_I[i][j] = acc

    for c in costs_I:
        print("costs:", c)
    
    return compute_da_algorithm(d, 0, len(d), k, costs_I)

def compute_da_algorithm(d, start_pos, length, k, costs_I, t_list=[]):
    if length < 2*k: 
        # t_list.insert(0, 0)
        return (costs_I[start_pos][start_pos + length - 1], t_list)
    costs = []
    for t in range(start_pos + k, start_pos + length - k + 1):
        costs.append([compute_da_algorithm(d, 0, t, k, costs_I, t_list)[0]+costs_I[t][-1], t])
    costs.append([costs_I[0][-1], 0])
    minimum = min(costs, key=lambda x: x[0])
    t_list.insert(0, minimum[1])
    return (minimum[0], t_list)

def compute_rec_da(d, k, t_list=[]):
    if len(d) < 2*k: 
        t_list.insert(0, 0)
        return (compute_I(d), t_list)
    costs = []
    for t in range(k, len(d)-k+1):
        costs.append([compute_rec_da(d[:t], k, t_list)[0]+compute_I(d[t:]), t])
    costs.append([compute_I(d), 0])
    minimum = min(costs, key=lambda x: x[0])
    t_list.insert(0, minimum[1])
    return (minimum[0], t_list)

def apply_I(d): 
    result = [d[0] for i in d]
    return result

def apply_da(d, k, t_list):
    for t in t_list: 
        if t == 0: return apply_I(d)
        else:
            result = apply_da(d[:t], k, t_list[1:])
            result.extend(apply_I(d[t:]))
            return result

def construct_graph(tab_index, anonymized_degree):
    graph = nx.Graph()
    if sum(anonymized_degree) % 2 == 1:
        return None
    
    if sum(anonymized_degree) > len(anonymized_degree)*(len(anonymized_degree)-1): 
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

    d_list = [
        [8,7,7,6,3,2,2,1],
        [3,2,2,2],
        [3,3,3,2,1],
        [4,3,3,2,1,1]
    ]
    k = 2

    for d in d_list:
        # print("d: ", d)
        # print(greedy_algorithm(d, k), "==", dp_slow_algorithm(d, k))

        # print("result: ", dp_algorithm(d, k))
        # print("--------")
        print("result slow: ", dp_slow_algorithm(d, k))
        print()

        # assert greedy_algorithm(d, k) == dp_slow_algorithm(d, k)
        # assert greedy_algorithm(d, k) == dp_algorithm(d,k)
 
