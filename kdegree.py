from random import randint
import numpy as np
import networkx as nx
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

def dp_algorithm(d, k): 
    if k == 1: raise ValueError("k == 1 means no anonymization!")

    #Da is the cost to k-anonymize a vector of degrees. The t variable stores, if there is one, where the first cut is made to anonymize.
    @dataclass
    class Da: 
        cost: float
        t: int

    # Having a bottom-top approach, the algorithm will calculate all the I costs, 
    # that is the costs to anonymize a vector puttin all the values equal. 
    # This is made for all the possible vectors. 
    def compute_all_I(d, k):  
        costs_I = [[float('inf') for _ in range(len(d))] for _ in range(len(d) - 1)]
        for i in range(len(d) - 1):
            acc = 0 
            for j in range(i + 1, len(d)):
                acc += d[i] - d[j]
                costs_I[i][j] = acc  
        return costs_I

    costs_I = compute_all_I(d, k) # Calculation of all I

    da_list = [Da(cost=float('inf'), t=-1) for _ in d] 

    # Calculation of all Da
    for i in range(len(d)):
        if i < 2*k - 1:
            da_list[i] = Da(cost=costs_I[0][i], t=0)
        else:
            t, cost = min(((t, da_list[t].cost + costs_I[t+1][i]) for t in range(k - 1, i - k + 1)), key= lambda x:x[1])
            if costs_I[0][i]  < cost:
                t = 0
                cost = costs_I[0][i]

            da_list[i] = Da(cost=cost, t=t)

    # Once all Da are in place, the algorithm takes the last one and see where cuts the vector. Each t is saved, since in an array. 
    # The array ends with a 0.
    t_list = []
    t = da_list[-1].t
    while t > 0:
        t_list.append(t)
        t = da_list[t].t
    t_list.append(0)

    # With the t, the algorithm cuts the vector and actually anonymize. 
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

def construct_graph(tab_index, anonymized_degree_list):
    anonymized_degree_list_temp = anonymized_degree_list.copy()
    graph = nx.Graph()
    if sum(anonymized_degree_list_temp) % 2 == 1:
        return None
    
    if sum(anonymized_degree_list_temp) > len(anonymized_degree_list_temp)*(len(anonymized_degree_list_temp)-1): 
        return None

    while True:
        if not all(di >= 0 for di in anonymized_degree_list_temp):
            return None
        if all(di == 0 for di in anonymized_degree_list_temp):
            return graph
        v = np.random.choice((np.where(np.array(anonymized_degree_list_temp) > 0))[0])
        dv = anonymized_degree_list_temp[v]
        anonymized_degree_list_temp[v] = 0
        for index in np.argsort(anonymized_degree_list_temp)[-dv:][::-1]:
            if index == v:
                return None
            if not graph.has_edge(tab_index[v], tab_index[index]):
                graph.add_edge(tab_index[v], tab_index[index])
                anonymized_degree_list_temp[index] = anonymized_degree_list_temp[index] - 1


if __name__ == "__main__":

    d_list = [
        [8,7,7,6,3,2,2,1],
        [3,2,2,2],
        [3,3,3,2,1],
        [4,3,3,2,1,1]
    ]

    results_k_2 = [
        [8,8,7,7,3,3,2,2],
        [3,3,2,2],
        [3,3,3,2,2],
        [4,4,3,3,1,1]
    ]

    k_list = [2,3]

    for k in k_list:
        for d in d_list:
            # print("d: ", d)
            # print(greedy_algorithm(d, k), "==", dp_slow_algorithm(d, k))

            # print("result: ", dp_algorithm(d, k))
            # print("--------")

            result = dp_algorithm(d, k)
            # result = dp_algorithm(d,k)
            # print("result slow: ",slow_result )
            print("result slow: ", result )

            # assert slow_result == results_k_2

            # assert greedy_algorithm(d, k) == dp_slow_algorithm(d, k)
            # assert greedy_algorithm(d, k) == dp_algorithm(d,k)
 
