#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import math
import time
import random
import argparse

import networkx as nx
import itertools
import utils


class DisjoinSet(object):

    def __init__(self, n):

        self.parent = list(range(n))
        self.rank = [0 for x in range(n)]

    def find(self, x):

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def merge(self, x, y):

        x_root = self.find(x)
        y_root = self.find(y)

        # Si estan en el mismo conjunto no hay que hacer nada
        if x_root == y_root:
            return

        # Si los rankings son distintos, nos quedamos con
        # el de la mayor raiz
        if self.rank[x_root] != self.rank[y_root]:

            max_root_rank = x_root if self.rank[x_root] > self.rank[y_root] else y_root
            self.parent[x_root] = max_root_rank
            self.parent[y_root] = max_root_rank

        else:

            self.parent[y_root] = x_root
            self.rank[x_root] += 1


def kruskal(graph):

    n = len(graph.nodes)
    tree = []
    priority_queue = sorted([(u, v, data['weight']) for u, v, data in graph.edges(data=True)], key=lambda x: x[2])
    disjoint_set = DisjoinSet(n)
    index = 0

    while len(tree) < n - 1 and index < len(priority_queue):

        (u, v, weight) = priority_queue[index]
        index += 1

        if disjoint_set.find(u) != disjoint_set.find(v):
            tree.append((u, v, weight))
            disjoint_set.merge(u, v)

    return tree


def kruskal_test(filename):
    
    graph = utils.load_undirected_and_unweighted_graph_from_txt_file(filename)
    tree = kruskal(graph)

    tree_graph = nx.Graph()
    tree_graph.add_nodes_from(graph.nodes)
    for u, v, weight in tree:
        tree_graph.add_edge(u, v, weight=weight)

    utils.plot_graph(tree_graph)


if __name__ == '__main__':
    
    filename = 'examples/graph0.txt'
    kruskal_test(filename)