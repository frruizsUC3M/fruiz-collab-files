#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import math
import time
import random
import argparse

import networkx as nx
import itertools


class NN(object):

    def __init__(self):

        super(NN, self).__init__()
        self.solutions = []

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

    def solve_complete(self, problem_graph):

        self.solutions = []

        best_solution = None
        best_cost = None
        nodes = problem_graph.nodes()

        for i in nodes:
            solution, cost = self.solve(problem_graph, i)
            self.solutions.append(solution)
            
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_solution = solution

        return best_solution, best_cost


    def solve(self, problem_graph, starting_node=None):

        solution = []
        cost = 0
        remaining_nodes = list(problem_graph.nodes())
        starting_node = starting_node if starting_node is not None else random.choice(remaining_nodes)

        solution.append(starting_node)
        remaining_nodes.remove(starting_node)

        remaining_nodes = set(remaining_nodes)

        while len(remaining_nodes) != 0:

            neighbors = problem_graph[solution[-1]]
            candidates = [(c, neighbors[c]['weight']) for c in neighbors.keys() if c in remaining_nodes]

            next_node = min(candidates, key = lambda c: c[1])
            solution.append(next_node[0])
            remaining_nodes.remove(next_node[0])
            cost += next_node[1]

        cost += problem_graph[solution[-1]][solution[0]]['weight']

        return solution, cost