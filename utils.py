#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import math
import time
import random
import argparse

import networkx as nx
import itertools
import matplotlib.pyplot as plt


def get_positions(filename):

    pos = {}

    with open(filename, 'r') as f:
        lines = f.readlines()[1:]
        for i, l in enumerate(lines):
            positions = l.split(' ')
            pos[i] = (float(positions[0]), float(positions[1]))

    return pos


def load_undirected_and_unweighted_graph_from_txt_file(filename):

    graph = nx.Graph()

    vertices = set([])
    edges = []

    with open(filename, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i != 0:
                edge = line.replace('\n', '').split(' ')
                for u in edge[:2]:
                    if not u in vertices:
                        vertices.add(int(u))
                edges.append(edge)

    graph.add_nodes_from(vertices)
    for edge in edges:
        graph.add_edge(int(edge[0]), int(edge[1]), weight=float(edge[2]))

    return graph


def verify_solution(solution, graph):

    for u in solution:
        for v in solution:
            if u != v:
                if not (u, v) in graph.edges(u):
                    return False

    return True


def generate_k_complete_graph_without_a_edge(k):

    graph = nx.complete_graph(k)
    graph.remove_edge(0, k-1)
    return graph


def convert_to_digraph(undirected_graph):

	G = nx.DiGraph()
	G.add_nodes_from(undirected_graph.nodes())

	for (u, v, data) in undirected_graph.edges(data=True):
		G.add_edge(u, v, weight=data['weight'] if 'weight' in data else 1)
		G.add_edge(v, u, weight=data['weight'] if 'weight' in data else 1)

	return G


def get_reduced_graph(solution, original_graph, directed=False):

    reduced_graph = nx.Graph() if not directed else nx.DiGraph()
    reduced_graph.add_nodes_from(solution)

    for i in range(len(solution)):

        u = solution[i]
        v = solution[(i+1)%len(solution)]

        weight = original_graph[u][v]['weight']
        reduced_graph.add_edge(u, v, weight=weight)

        if directed:
        	reduced_graph.add_edge(v, u, weight=weight)

    return reduced_graph


def euclidean_distance(point1, point2):

    return math.sqrt( (point1.x - point2.x)**2 + (point1.y - point2.y)**2 )


def plot_graph(G, positions=None):

    edges = G.edges(data=True)
    pos = nx.spring_layout(G) if positions is None else positions  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=16)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1)

    plt.axis('off')
    plt.show()


def plot_graph_with_edge_selection(G, solution):

    edges = G.edges(data=True)
    pos = nx.spring_layout(G)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=16)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=solution, edge_color='r', width=3)
    nx.draw_networkx_edges(G, pos, edgelist=[e for e in G.edges() if e not in solution], width=0.5)

    plt.axis('off')
    plt.show()


def plot_graph_with_solution(G, solution):

    edges = G.edges(data=True)
    pos = nx.spring_layout(G)  # positions for all nodes

    # nodes
    G_vertex_selected = nx.Graph()
    G_vertex_selected.add_nodes_from(solution)
    nx.draw_networkx_nodes(G_vertex_selected, pos, node_size=16, node_color='b')

    G_not_vertex_selected = nx.Graph()
    G_not_vertex_selected.add_nodes_from([n for n in G.nodes() if n not in solution])
    nx.draw_networkx_nodes(G_not_vertex_selected, pos, node_size=16, node_color='r')

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1)

    plt.axis('off')
    plt.show()

