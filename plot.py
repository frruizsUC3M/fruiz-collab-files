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

        plt.plot(df['cost'], marker='.')
        legends.append(f'T(n) {algorithm_name}')

    # plt.plot([0.0000002*i**3 for i in range(6, 432)])
    # legends.append('an^3')

    # plt.plot([0.00000011*i**3 for i in range(6, 432)])
    # legends.append('bn^3')

    # plt.plot([0.0000007*math.log(i)*i**2 for i in range(6, 432)])
    # legends.append('c n^2 log n')


    plt.title(f'Time evolution in reduction approach and exact algo\'s')
    plt.ylabel('Time (s)')
    plt.xlabel(f'Problem size (n, {xlabel})')
    plt.legend(legends, loc='upper left')
    plt.savefig(f'{output_filename}_evolution.png')
    plt.show()


def load_experiments_data(path, algorithm_names):

    dfs = []
    for algo in algorithm_names:
        filename = os.path.join(path, f'results_{algo}.csv')
        df = pd.read_csv(filename, header=0)
        dfs.append(df)

    return dfs


if __name__ == '__main__':

    # algos = ['mst_aproximation', 'christofides_aproximation', 'nn_aproximation']
    # algos = ['reduced_edges_aproximation', 'reduced_edges_2_aproximation', 'planar_aproximation', 'planar_aproximation_2', 'exact', 'christofides_aproximation']
    algos = ['exact', 'local_search_augmented', 'gls']

    dfs = load_experiments_data('results_experiment_big', algos)
    plot_results_combine(dfs, 'results_experiment_big')