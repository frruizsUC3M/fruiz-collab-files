# https://python-mip.readthedocs.io/en/latest/custom.html
# Edited by Fran Ruiz

from typing import List, Tuple
from random import seed, randint
from itertools import product
import networkx as nx
from mip import Model, xsum, BINARY, minimize, ConstrsGenerator, CutPool

import math
import time

import argparse


class SubTourCutGenerator(ConstrsGenerator):

    def __init__(self, Fl: List[Tuple[int, int]], x_, V_, graph):

        self.F, self.x, self.V = Fl, x_, V_
        self.graph = graph

    def generate_constrs(self, model: Model):

        xf, V_, cp, G = model.translate(self.x), self.V, CutPool(), nx.DiGraph()

        # valid edges
        for (u, v) in self.graph.edges():
            try:
                edge = xf[u][v]
                G.add_edge(u, v, capacity=xf[u][v].x)
            except:
                pass

        for (u, v) in self.F:
            try:
                val, (S, NS) = nx.minimum_cut(G, u, v)
                if val <= 0.99:
                    aInS = [(xf[i][j], xf[i][j].x)
                            for (i, j) in product(V_, V_) if i != j and xf[i][j] and i in S and j in S]
                    if sum(f for v, f in aInS) >= (len(S)-1)+1e-4:
                        cut = xsum(1.0*v for v, fm in aInS) <= len(S)-1
                        cp.add(cut)
                        if len(cp.cuts) > 256:
                            for cut in cp.cuts:
                                model += cut
                            return
            except nx.NetworkXError:
                pass
        for cut in cp.cuts:
            model += cut


class TSP(object):

    """docstring for TSP"""
    def __init__(self):

        super(TSP, self).__init__()
        self.model = Model()
        self.graph = None

    def load_problem(self, graph):

        self.graph = graph
        return self

    def get_farthest_point_list(self):

        # computing farthest point for each point, these will be checked first for
        # isolated subtours

        F = []
        for i in self.graph.nodes():
            P, D = nx.dijkstra_predecessor_and_distance(self.graph, source=i)
            DS = list(D.items())
            farthest_point = max(DS, key=lambda x: x[1])[0]
            F.append((i, farthest_point))

        return F

    def add_variables(self):

        # binary variables indicating if arc (i,j) is used on the route or not
        self.x = { (i, j): self.model.add_var(var_type=BINARY) for i, j in self.Arcs }

        # continuous variable to prevent subtours: each city will have a
        # different sequential id in the planned route except the first one
        self.y = [self.model.add_var() for i in self.V]

    def add_constraints(self):

        # constraint : leave each city only once
        for i in self.V:
            self.model += xsum(self.x[(i, j)] for j in self.graph.neighbors(i)) == 1

        # constraint : enter each city only once
        for i in self.V:
            self.model += xsum(self.x[(j, i)] for j in self.graph.neighbors(i)) == 1

        # (weak) subtour elimination
        # subtour elimination
        for (i, j) in self.Arcs:
            if i != 0 and j != 0:
                self.model += self.y[i] - (self.n+1)*self.x[(i, j)] >= self.y[j]-self.n

        # no subtours of size 2
        for (i, j) in self.Arcs:
            self.model += self.x[(i, j)] + self.x[(j, i)] <= 1

    def build(self, min_obj_value=None, max_n_solutions=None):

        self.V = set(self.graph.nodes())
        self.n = len(self.V)
        seed(0)
        self.Arcs = self.graph.edges()

        self.add_variables()
        self.add_constraints()

        if min_obj_value is not None:
            constr = xsum(self.graph[i][j]['weight']*self.x[(i, j)] for (i, j) in self.Arcs) > min_obj_value
            self.model +=constr

        # objective function: minimize the distance
        self.model.objective = minimize(xsum(self.graph[i][j]['weight']*self.x[(i, j)] for (i, j) in self.Arcs))


        self.F = self.get_farthest_point_list()
        self.model.cuts_generator = SubTourCutGenerator(self.F, self.x, self.V, self.graph)

        return self

    def solve(self, max_seconds=None):

        if max_seconds is None:
            self.model.optimize()
        else:
            self.model.optimize(max_seconds=max_seconds)

        self.V = list(self.V)

        solution = [self.V[0]]
        while len(solution) != len(self.V):
            last_vertice = solution[-1]
            for vertice2 in self.V:
                try:
                    if self.x[(last_vertice, vertice2)].x == 1.0:
                        solution.append(vertice2)
                        break
                except:
                    pass

        return solution, self.model.objective_value


def tsp_mip_cutting_planes(problem):

    start_time = time.time()
    
    tsp = TSP(problem)
    tsp.load_problem()
    tsp.build()
    solution, cost = tsp.solve()

    print(f'Solution: {solution}')
    print(f'Cost: {cost}')


    end_time = time.time()
    total_time = round((end_time - start_time) * 1000, 4)

    print(f'Time: {total_time}')

    return solution, cost, total_time


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="B&C MIP Algorithm")
    parser.add_argument(
        "-f",
        "--File",
        type=str,
        help="Path to the file containing the data",
        required=True,
    )
    args = parser.parse_args()
    
    tsp_mip_cutting_planes(args.File)