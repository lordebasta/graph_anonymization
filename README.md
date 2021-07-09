# Graph Anonymization
This project regards the k-degree anonymity research by Liu and Terzi. It is a python implementation of the algorithms presented in the paper.

## Repository structure
k-degree-anonymity paper.pdf: it is the full research about the graph anonymization which explains in the detail the idea and the implementation of the algorithms.

Dataset folder: the dataset folder contains all the graph, in .csv format, used for running and testing the algorithms.

k-degree.ipynb: it is a jupyter notebook that shows and briefly explains how the graph anonymization works. Running it, it is possible to see how the algorithms should be used and the differences between the slow and the fast one.

## Breaf explanation of the idea
The aim of this research is to solve an existing problem: anonymizing a graph by simply removing the identitites from the nodes is not a good solution for maintaining the privacy, since it is possible to deduce an individual in the network studying the connections between the nodes and the edges' nodes.
The solution consists of minimally modifing the graph maintainig the privacy of each individual involved.
