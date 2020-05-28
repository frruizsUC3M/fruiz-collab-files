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


class Christofides(object):

    def __init__(self):

        super(Christofides, self).__init__()

    def get_mst(self, problem_graph):

        mst = nx.algorithms.tree.mst.minimum_spanning_tree(problem_graph)
        return mst

    def build_graph_with_odd_vertex(self, problem_graph, mst):
        
        odd_vertex = [n for n in mst.nodes if mst.degree[n] % 2 == 1]
        odd_graph = nx.Graph()
        odd_graph.add_nodes_from(odd_vertex)

        max_weight = max(problem_graph.edges(data=True), key = lambda x: x[2]['weight'])[2]['weight'] + 1e-04
        for u in odd_graph.nodes:
            for v in odd_graph.nodes:
                if u != v:
                    odd_graph.add_edge(u, v, weight=max_weight - problem_graph[u][v]['weight'])

        return odd_graph

    def join_mst_and_matching_edges(self, problem_graph, mst, matching_edges):

        mst_extended = nx.MultiGraph()
        mst_extended.add_nodes_from(problem_graph.nodes)

        for u, v, data in mst.edges(data=True):
            mst_extended.add_edge(u, v, weight=data['weight'])

        for u, v in matching_edges:
            mst_extended.add_edge(u, v, weight=problem_graph[u][v]['weight'])

        return mst_extended

    def fast_hamiltonian_circuit_from_euler_circuit(self, walk):

        solution = [walk[0][0]]
        in_solution = set(solution)

        for u, v in walk:
            if not v in in_solution:
                solution.append(v)
                in_solution.add(v)

        return solution

    def solve(self, problem_graph):

        solution = []
        cost = 0

        mst = self.get_mst(problem_graph)
        odd_graph = self.build_graph_with_odd_vertex(problem_graph, mst)

        matching_edges = nx.algorithms.matching.max_weight_matching(odd_graph)
        mst_extended = self.join_mst_and_matching_edges(problem_graph, mst, matching_edges)

        walk = list(nx.eulerian_circuit(mst_extended))
        solution = self.fast_hamiltonian_circuit_from_euler_circuit(walk)

        for i in range(len(solution)):

            u = solution[i]
            v = solution[(i+1)%len(solution)]
            
            weight = problem_graph[u][v]['weight']

            cost += weight

        return solution, cost


if __name__ == '__main__':
    
    filename = 'examples/ulysses16.json'
    original_problem_graph = TSPProblemParser().parse(filename)
    solution, cost = Christofides().solve(original_problem_graph)
    print(cost)