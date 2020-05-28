#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import math
import time
import random
import argparse

from collections import namedtuple
import networkx as nx
import itertools

from utils import euclidean_distance


Point = namedtuple("Point", ['x', 'y'])
NNSolution = namedtuple("NNSolution", ['solution', 'cost'])


class ProblemParser(object):

    def __init__(self):
        super(ProblemParser, self).__init__()

    def parse(self, filename):

        return nx.DiGraph()


class TSPProblemParser(ProblemParser):

    def __init__(self):

        super(ProblemParser, self).__init__()

    def read_points(self, filename):

        with open(filename, 'r') as f:
            input_data = f.read()

        lines = input_data.split('\n')

        node_count = int(lines[0])

        points = []
        for i in range(1, node_count+1):
            line = lines[i]
            parts = line.split()
            points.append(Point(float(parts[0]), float(parts[1])))

        return points

    def add_nodes(self, graph, points):

        for i, point in enumerate(points):
            graph.add_node(i)

        return graph

    def add_edges(self, graph, points, directed=True):

        nodes = graph.nodes()
        it = itertools.combinations(nodes, 2)

        for (u, v) in it:
            d = euclidean_distance(points[u], points[v])
            graph.add_edge(u, v, weight=d)
            if directed:
                graph.add_edge(v, u, weight=d)

        return graph
    
    def parse(self, filename):

        points = self.read_points(filename)
        graph = nx.Graph()
        graph = self.add_nodes(graph, points)
        graph = self.add_edges(graph, points, directed=False)

        return graph

    def parseDirected(self, filename):

        points = self.read_points(filename)
        graph = nx.DiGraph()
        graph = self.add_nodes(graph, points)
        graph = self.add_edges(graph, points, directed=True)

        return graph
