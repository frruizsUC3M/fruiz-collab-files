import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os

import utils
from problem_parser import TSPProblemParser


def get_size_column(x):

    graph = TSPProblemParser().parse(x)
    return len(graph.nodes())


def plot_results_combine(dfs, output_filename, size_func=get_size_column, xlabel='#vertex'):

    legends = []

    for df in dfs:

        algorithm_name = df['func_name'][0]
        experiment_name = df['name'][0]

        df['size'] = df['filename'].apply(size_func)
        df = df[['size', 'cost']].groupby(['size']).max().reset_index().sort_values(by=['size']).set_index('size')

        plt.plot(df['cost'])
        legends.append(f'T(n) {algorithm_name}')


    plt.title(f'cost evolution in experiment {experiment_name}.')
    plt.ylabel('cost')
    plt.xlabel(f'Problem size (n, {xlabel})')
    plt.legend(legends, loc='upper left')
    plt.savefig(f'{output_filename}_cost_evolution.png')
    plt.show()


def load_experiments_data(path, algorithm_names):

    dfs = []
    for algo in algorithm_names:
        filename = os.path.join(path, f'results_{algo}.csv')
        df = pd.read_csv(filename, header=0)
        dfs.append(df)

    return dfs


if __name__ == '__main__':
    
    # algos = ['dfs', 'dfs_complete', 'dfs_optimal', 'dfs_optimal_complete',
    # 'warshall', 'floyd_warshall', 'tarjan', 'kosaraju']

    # algos = ['dfs', 'tarjan', 'kosaraju', 'warshall', 'dfs_optimal']

    # algos = ['dfs_complete', 'dfs_optimal_complete', 'warshall', 'floyd_warshall']

    # algos = ['clique_mip', 'clique_iterated_local_search']
    # algos = ['clique_mip', 'clique_brute_force',
    # 'clique_brute_force_discarding', 'clique_iterated_local_search']

    algos = ['mst_aproximation', 'christofides_aproximation', 'nn_aproximation']
    algos = ['reduced_edges_aproximation', 'reduced_edges_2_aproximation', 'planar_aproximation', 'planar_aproximation_2', 'exact', 'local_search_augmented']
    algos = ['exact', 'local_search_augmented', 'gls']

    dfs = load_experiments_data('results_experiment_big', algos)
    plot_results_combine(dfs, 'results_experiment_big')