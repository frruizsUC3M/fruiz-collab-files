#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import math
import time
import random
import argparse

import networkx as nx
import itertools


class PMFG(object):

    def __init__(self):

        super(PMFG, self).__init__()

    def sort_graph_edges(self, G):

        sorted_edges = []
        for source, dest, data in sorted(G.edges(data=True),
                                         key=lambda x: x[2]['weight']):

            sorted_edges.append({'source': source,
                                 'dest': dest,
                                 'weight': data['weight']})
            
        return sorted_edges

    def fix_PMFG(self, pmfg, original_graph):

        if len(pmfg.nodes()) == len(original_graph.nodes()):
            print('Not need to fix anything.')
            return pmfg

        for node in original_graph.nodes():
            if node not in pmfg.nodes():
                min_edge = min(original_graph.edges(node, data=True), key=lambda x: x[2]['weight'])
                pmfg.add_edge(min_edge[0], min_edge[1], weight=min_edge[2]['weight'])

        return pmfg

    def compute_PMFG(self, sorted_edges, original_graph):

        PMFG = nx.Graph()
        nb_nodes = len(original_graph.nodes)

        for edge in sorted_edges:

            PMFG.add_edge(edge['source'], edge['dest'], weight=edge['weight'])
            
            if not nx.algorithms.planarity.check_planarity(PMFG):
                PMFG.remove_edge(edge['source'], edge['dest'])
                
            if len(PMFG.edges()) == 3*(nb_nodes-2):
                break

        PMFG = self.fix_PMFG(PMFG, original_graph)
        
        return PMFG

    def get_reduced_graph(self, original_graph):

        sorted_edges = self.sort_graph_edges(original_graph)
        reduced_graph = self.compute_PMFG(sorted_edges, original_graph)

        return reduced_graph