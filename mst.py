#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import math
import time
import random
import argparse

import networkx as nx
import itertools

from problem_parser import TSPProblemParser


class MST(object):

    def __init__(self):

        super(MST, self).__init__()

    def get_reduced_graph(self, solution, original_graph):

        reduced_graph = nx.DiGraph()
        reduced_graph.add_nodes_from(solution)

        for i in range(len(solution)):

            u = solution[i]
            v = solution[(i+1)%len(solution)]

            weight = original_graph[u][v]['weight']
            reduced_graph.add_edge(u, v, weight=weight)
            reduced_graph.add_edge(v, u, weight=weight)

        return reduced_graph

    def solve(self, problem_graph):

        solution = []
        cost = 0

        mst = nx.algorithms.tree.mst.minimum_spanning_tree(problem_graph)
        solution = list(nx.dfs_preorder_nodes(mst, source=0))

        for i in range(len(solution)):

            u = solution[i]
            v = solution[(i+1)%len(solution)]

            weight = problem_graph[u][v]['weight']
            cost += weight

        return solution, cost


if __name__ == '__main__':
    
    filename = 'examples/ulysses16.json'
    original_problem_graph = TSPProblemParser().parse(filename)
    solution, cost = MST().solve(original_problem_graph)
    print(cost)