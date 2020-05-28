#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import math
import time
import random
import argparse

import networkx as nx
from collections import namedtuple
import matplotlib.pyplot as plt
import itertools

import csv
import pandas as pd

import tsp_mip_cutting_planes
import tsp_mip_cutting_planes_fixed

from ls import LocalSearchAugmented
from nn import NN
from mst import MST
from christofides import Christofides
from pmfg import PMFG
from problem_parser import TSPProblemParser
from utils import plot_graph, euclidean_distance, get_reduced_graph, convert_to_digraph, get_positions


Point = namedtuple("Point", ['x', 'y'])
NNSolution = namedtuple("NNSolution", ['solution', 'cost'])


def local_search_augmented(filename, batch_size=8, n_iters=30, max_attemps=4, max_seconds=None):

    original_problem_graph = TSPProblemParser().parseDirected(filename)
    solution, cost = LocalSearchAugmented().solve(original_problem_graph, batch_size, n_iters, max_attemps, max_seconds)
    print(f'solution = {solution}, cost = {cost}')
    return solution, cost


def exact(filename, max_seconds=None):

    original_problem_graph = TSPProblemParser().parseDirected(filename)
    solution, cost = tsp_mip_cutting_planes_fixed.TSP().load_problem(original_problem_graph,
        original_problem_graph).build().solve(max_seconds)
    print(f'solution = {solution}, cost = {cost}')
    return solution, cost


def mst_aproximation(filename):

    original_problem_graph = TSPProblemParser().parse(filename)
    solution, cost = MST().solve(original_problem_graph)
    # print(f'solution = {solution}, cost = {cost}')
    return solution, cost


def christofides_aproximation(filename):

    original_problem_graph = TSPProblemParser().parse(filename)
    solution, cost = Christofides().solve(original_problem_graph)
    # print(f'solution = {solution}, cost = {cost}')
    return solution, cost


def nn_aproximation(filename):

    original_problem_graph = TSPProblemParser().parse(filename)
    solution, cost = NN().solve_complete(original_problem_graph)
    # print(f'solution = {solution}, cost = {cost}')
    return solution, cost


def reduced_edges_aproximation(filename, max_seconds=None):

    original_problem_graph = TSPProblemParser().parse(filename)
    solution_mst, cost = MST().solve(original_problem_graph)

    solution_nn, cost = NN().solve_complete(original_problem_graph)

    reduced_graph = nx.operators.compose(get_reduced_graph(solution_mst, original_problem_graph, True),
        get_reduced_graph(solution_nn, original_problem_graph, True))
    graph = convert_to_digraph(original_problem_graph)

    solution, cost = tsp_mip_cutting_planes_fixed.TSP().load_problem(graph, reduced_graph).build().solve(max_seconds)
    print(f'solution = {solution}, cost = {cost}')
    return solution, cost


def reduced_edges_2_aproximation(filename, max_seconds=None):

    original_problem_graph = TSPProblemParser().parse(filename)
    solution_mst, cost = MST().solve(original_problem_graph)

    nn_algo = NN()
    solution_nn, cost = nn_algo.solve_complete(original_problem_graph)

    all_nn_graph = get_reduced_graph(solution_nn, original_problem_graph, True)
    for s in nn_algo.solutions:
        all_nn_graph = nx.operators.compose(all_nn_graph, get_reduced_graph(s, original_problem_graph, True))

    reduced_graph = nx.operators.compose(all_nn_graph, get_reduced_graph(solution_mst, original_problem_graph, True))

    graph = convert_to_digraph(original_problem_graph)

    solution, cost = tsp_mip_cutting_planes_fixed.TSP().load_problem(graph, reduced_graph).build().solve(max_seconds)
    print(f'solution = {solution}, cost = {cost}')
    return solution, cost


def planar_aproximation(filename, max_seconds=None):

    original_problem_graph = TSPProblemParser().parse(filename)
    planar_graph = PMFG().get_reduced_graph(original_problem_graph)

    solution_nn, cost = NN().solve_complete(original_problem_graph)
    tour_graph = get_reduced_graph(solution_nn, original_problem_graph, True)

    planar_graph = convert_to_digraph(planar_graph)
    graph = convert_to_digraph(original_problem_graph)

    reduced_graph = nx.operators.compose(planar_graph, tour_graph)
    
    solution, cost = tsp_mip_cutting_planes.TSP().load_problem(reduced_graph).build().solve(max_seconds)
    print(f'solution = {solution}, cost = {cost}')
    return solution, cost


def planar_aproximation_2(filename, max_seconds=None):

    original_problem_graph = TSPProblemParser().parse(filename)
    planar_graph = PMFG().get_reduced_graph(original_problem_graph)

    nn_algo = NN()
    solution_nn, cost = nn_algo.solve_complete(original_problem_graph)

    all_nn_graph = get_reduced_graph(solution_nn, original_problem_graph, True)
    for s in nn_algo.solutions:
        all_nn_graph = nx.operators.compose(all_nn_graph, get_reduced_graph(s, original_problem_graph, True))

    planar_graph = convert_to_digraph(planar_graph)
    graph = convert_to_digraph(original_problem_graph)

    reduced_graph = nx.operators.compose(planar_graph, all_nn_graph)
    
    solution, cost = tsp_mip_cutting_planes_fixed.TSP().load_problem(graph, reduced_graph).build().solve(max_seconds)
    print(f'solution = {solution}, cost = {cost}')
    return solution, cost


def gls(filename, alpha=0.4, max_iterations=10000, max_attemps=4):

    cmd = f'gls.exe -f={filename} -alpha={alpha} -maxIterations={max_iterations} -maxAttemps={max_attemps}'
    os.system(cmd)

    with open('solution.txt', 'r') as f:
        cost = float(f.readline())
        solution = [int(i) for i in f.readline().split(' ')[:-1]]

    print(f'solution = {solution}, cost = {cost}')
    return solution, cost


def plot_all_nn_solutions(filename):

    original_problem_graph = TSPProblemParser().parse(filename)
    nn_algo = NN()
    solution, cost = nn_algo.solve_complete(original_problem_graph)
    print(f'solution = {solution}, cost = {cost}')

    graph = get_reduced_graph(solution, original_problem_graph, False)
    for s in nn_algo.solutions:
        graph = nx.operators.compose(graph, get_reduced_graph(s, original_problem_graph, False))

    pos = get_positions(filename)
    print(pos)
    plot_graph(original_problem_graph, positions=pos)
    plot_graph(get_reduced_graph(solution, original_problem_graph, False), positions=pos)
    plot_graph(graph, positions=pos)

    return solution, cost


if __name__ == '__main__':
    
    # plot_all_nn_solutions('examples/ulysses22.json')
    reduced_edges_aproximation('examples/ulysses22.json')
    reduced_edges_2_aproximation('examples/ulysses22.json')