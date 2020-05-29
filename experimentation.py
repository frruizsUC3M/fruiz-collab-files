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

from algorithms import exact
from algorithms import local_search_augmented
from algorithms import mst_aproximation
from algorithms import christofides_aproximation
from algorithms import nn_aproximation
from algorithms import reduced_edges_aproximation
from algorithms import reduced_edges_2_aproximation
from algorithms import planar_aproximation
from algorithms import planar_aproximation_2
from algorithms import gls


class Experimenter(object):

    def __init__(self, name='default', size_problem_func=None, n_repeats=1):

        super(Experimenter, self).__init__()
        self.name = name
        self.history = []
        self.solutions = []
        self.size_problem_func = size_problem_func
        self.n_repeats = n_repeats

    def run_multiple_from_parameters(self, func, func_name, args):

        names = list(args.keys())
        values = [args[n] for n in names]

        all_combinations = itertools.product(*values)

        for combination in all_combinations:
            print(f'Launching {combination} ...')
            arg = {names[i]: combination[i] for i in range(len(combination))}
            self.run_one(func, func_name, arg)

        return self

    def run_one(self, func, func_name, args):

        times = []
        
        for i in range(self.n_repeats):
            start_time = time.time()
            solution, cost = func(**args)
            end_time = time.time()
            total_time = round((end_time - start_time), 4)
            times.append(total_time)

        total_time_mean = sum(times)/self.n_repeats

        data = dict(args)
        data['size'] = len(solution) if self.size_problem_func is None else self.size_problem_func(**args)
        data['cost'] = cost
        data['time'] = total_time_mean
        data['index'] = len(self.history)
        data['name'] = self.name
        data['func_name'] = func_name

        self.history.append(data)
        self.solutions.append(solution)

        return self

    def generate_csv(self, output_filename):

        columns = set([])
        for data in self.history:
            for k in data:
                columns.add(k)
        columns = list(columns)

        df_data = {c: [] for c in columns}
        for data in self.history:
            for c in df_data:
                if c in data:
                    df_data[c].append(data[c])
                else:
                    df_data[c].append(None)

        df = pd.DataFrame(df_data)
        df.to_csv(output_filename, quoting=csv.QUOTE_NONNUMERIC)

    def save_solutions(self, output_dir):

        for i in range(len(solutions)):

            data = dict(self.history[i])
            data['solution'] = self.solutions[i]

            filename = os.path.join(output_dir, self.name, 'solution{i}.json')
            with open(filename, 'w') as f:
                f.write(json.dumps(data))


def solution_size_func(filename, s=None, t=None):

    return 0


def get_filenames_experiment(experiment):

    root = os.path.join('examples', experiment)
    filenames = []

    for filename in os.listdir(root):
        print(filename)
        filenames.append(os.path.join(root, filename))

    return filenames


def create_results_folder(experiment):

    try:
        os.mkdir(f'results_experiment_{experiment}')
    except:
        pass


def get_args_algorithm(algo, filenames):

    if algo == 'mst_aproximation' or algo == 'nn_aproximation':

        return {

            'filename': filenames
        }

    lal = ['exact', 'reduced_edges_aproximation', 'reduced_edges_2_ aproximation',
    'planar_aproximation', 'planar_aproximation_2']

    if algo in lal:

        return {

            'filename': filenames,
            'max_seconds': [600]
        }

    if algo == 'local_search_augmented':

        return {

            'filename': filenames,
            'max_seconds': [120],
            'max_attemps': [2, 4],
            'batch_size': [16, 64],
            'n_iters': [12]
        }

    if algo == 'gls':

        return {

            'filename': filenames,
            'alpha': [0.7],
            'max_iterations': [50000],
            'max_attemps': [4]
            # 'alpha': [0.4],
            # 'max_iterations': [100000],
            # 'max_attemps': [32]
        }

    return {

            'filename': filenames
        }


def tsp_experimenter(experiment):

    algorithms = ['mst_aproximation', 'nn_aproximation', 'christofides_aproximation',
    'reduced_edges_aproximation', 'reduced_edges_2_aproximation', 'planar_aproximation', 'planar_aproximation_2', 'exact']

    # algorithms = ['mst_aproximation', 'christofides_aproximation', 'nn_aproximation']

    algorithms = ['gls']

    filenames = get_filenames_experiment(experiment)

    for algo in algorithms:

        filename = os.path.join(f'results_experiment_{experiment}', f'results_{algo}.csv')
        create_results_folder(experiment)

        Experimenter(experiment, size_problem_func=None, n_repeats=1).\
            run_multiple_from_parameters(eval(algo), algo, get_args_algorithm(algo, filenames)).\
                generate_csv(filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="TSP Experimenter")
    parser.add_argument(
        "-e",
        "--Experiment",
        type=str,
        help="Folder name in examples with TSP instances to test",
        required=True,
    )
    args = parser.parse_args()

    tsp_experimenter(args.Experiment)