#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import math
import time
import random
import argparse

import networkx as nx
from collections import namedtuple
import matplotlib.pyplot as plt
import itertools

import tsp_mip_cutting_planes_fixed

from nn import NN
from pmfg import PMFG
from utils import plot_graph, euclidean_distance, get_reduced_graph, convert_to_digraph


Point = namedtuple("Point", ['x', 'y'])
NNSolution = namedtuple("NNSolution", ['solution', 'cost'])


class LocalSearchAugmented(object):

    def __init__(self):

        super(LocalSearchAugmented, self).__init__()

    def solve_all_nns(self, nodes, original_graph):
        """ Calcula todas las soluciones obtenidas a partir de resolver el TSP original
        mediante el algoritmo NN."""

        nn_solutions = []

        for i in nodes:
            solution, cost = NN().solve(original_graph, starting_node=i)
            nn_solutions.append(NNSolution(solution, cost))

        return nn_solutions

    def get_initial_reduced_graph(self, nn_solutions, original_graph):
        """ Devuelve la mejor solucion encontrada mediante el algoritmo NN."""

        nn_solution = min(nn_solutions, key=lambda x: x.cost)
        return NN().get_reduced_graph(nn_solution.solution, original_graph), nn_solution

    def merge_graphs(self, all_graphs):
        """ Devuelve una unica lista de aristas con todas las aristas de la lista de grafos recibida. """

        sorted_edges = []
        for graph in all_graphs:
            edges = graph.edges(data=True)
            for edge in edges:
                tedge = (min(edge[0], edge[1]), max(edge[0], edge[1]), edge[2])
                if not tedge in sorted_edges:
                    sorted_edges.append(edge)

        return sorted_edges

    def solve(self, original_graph, batch_size=8, n_iters=30, max_attemps=4, max_seconds=None):

        nodes = list(original_graph.nodes())

        # calcula todos los NN y obtiene la mejor solucion
        nn_solutions = self.solve_all_nns(nodes, original_graph)
        initial_reduced_graph, nn_solution = self.get_initial_reduced_graph(nn_solutions, original_graph)

        best_cost = nn_solution.cost
        best_solution = nn_solution.solution

        # calcula el PMFG del grafo original
        last_n_edges = len(initial_reduced_graph.edges())
        nn_solutions = sorted(nn_solutions, key=lambda x: x.cost)
        PMFG_graph = PMFG().get_reduced_graph(original_graph)

        # obtiene todas las aristas de la union de todos los grafos resultantes del PMFG y NN
        all_graphs = [PMFG_graph]
        for solution in nn_solutions:
            graph = NN().get_reduced_graph(solution.solution, original_graph)
            all_graphs.append(graph)
        
        sorted_edges = self.merge_graphs(all_graphs)

        # damos prioridad a la aristas de menor peso
        sorted_edges = sorted(sorted_edges, key=lambda x: (x[2]['weight'], x[0], x[1]))

        it = 0
        attemps = 0
        total_edges = len(initial_reduced_graph.edges)

        while it < n_iters and (it+1)*2*batch_size < len(sorted_edges) and attemps < max_attemps:

            # creamos un grafo con las nuevas aristas seleccionadas
            next_edges = sorted_edges[it*2*batch_size: (it+1)*2*batch_size]
            graph = nx.DiGraph()
            graph.add_nodes_from(initial_reduced_graph.nodes())

            for edge in next_edges:
                graph.add_edge(edge[0], edge[1], weight=edge[2]['weight'])
                graph.add_edge(edge[1], edge[0], weight=edge[2]['weight'])

            # creamos n grafo temporal a partir del actual y el que tiene las nuevas aristas
            temp_reduced_graph = nx.operators.compose(initial_reduced_graph, graph)

            # resolvemos usando el grafo temporal
            solution, cost = \
                tsp_mip_cutting_planes_fixed.TSP().load_problem(original_graph, temp_reduced_graph).build().solve(max_seconds)

            if solution is None:
                it += 1
                attemps += 1
                continue

            # si mejoramos la solucion, mantendremos las nuevas aristas añadidas
            if best_cost is None or cost < best_cost:

                best_cost = cost
                best_solution = solution
                initial_reduced_graph = temp_reduced_graph
                attemps = 0

            it += 1
            attemps += 1

            num_new_edges = len(initial_reduced_graph.edges()) - last_n_edges
            last_n_edges = len(initial_reduced_graph.edges())

        return best_solution, best_cost