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


def c_merge(d, d1, k):
    c_merge_cost = d1 - d[k] + compute_I(d[k+1:min(len(d), 2*k)])
    return c_merge_cost


def c_new(d, k):
    t = d[k:min(len(d), 2*k-1)]
    c_new_cost = compute_I(t)
    return c_new_cost


def greedy_rec_algorithm(array_degrees, k_degree, pos_init, extension):
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
            greedy_rec_algorithm(array_degrees, k_degree, pos_init + extension, k_degree)
        else:
            greedy_rec_algorithm(array_degrees, k_degree, pos_init, extension + 1)



def dp_graph_anonymization():
    # TODO to complete
    return


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
    # take the arguments k and the dataset
    k_degree = int(sys.argv[1])
    file_graph = sys.argv[2]

    G = nx.Graph()
    
    if os.path.exists(file_graph): 
        # if file exist
        with open(file_graph) as f:
            content = f.readlines()
        # read each line
        content = [x.strip() for x in content]
        for line in content:
            # split name inside each line
            names = line.split(",")
            start_node = names[0]
            if start_node not in G:
                G.add_node(start_node)
            for index in range(1, len(names)):
                node_to_add = names[index]
                if node_to_add not in G:
                    G.add_node(node_to_add)
                G.add_edge(start_node, node_to_add)

    nx.draw(G)

    # Degree arrays preparation
    d = [x[1] for x in G.degree()]
    array_index = np.argsort(d)[::-1]
    array_degrees = np.sort(d)[::-1]
    print("Array of degrees (d) : {}".format(d))
    print("Array of degrees sorted (array_degrees) : {}".format(array_degrees))
    array_degrees_greedy = array_degrees
    # TODO insert here the code
    greedy_rec_algorithm(array_degrees_greedy, k_degree, 0, k_degree)

    graph_greedy = construct_graph(array_index, array_degrees_greedy)
    print("graph_greedy_nodes: ", end="")
    print(graph_greedy.nodes)
    print("graph_greedy_nodes sorted: ", end="")
    print(np.sort(graph_greedy.nodes))
