import numpy as np
import networkx as nx

from data import *
from utils.readable_df import *
from utils.weight_functions import *
from utils.financial_measures import calculate_measures
from utils.measures import *


class SplexSolver:
    def __init__(self, G, s=2):
        self.G = G
        self.s = s
        self.N = len(G.nodes())
        self.max = 0
        self.maxi = [0 for i in range(self.N)]
        self.res = []
        self.order = []


    def w(self, X):
        res = 0
        for i in X:
            res += self.G.nodes[i]["weight"]
        return res


    def is_splex(self, X):
        Gp = self.G.subgraph(X)
        for x in Gp.nodes():
            if Gp.degree(x) < len(Gp.nodes()) - self.s:
                return False
        return True


    def min_order(self, X):
        mini, m = self.N, self.N
        for v in X:
            if self.order.index(v) < mini:
                mini = self.order.index(v)
                m = v
        return mini, m


    def find_max(self, C, P):
        if len(C) == 0:
            if self.w(P) > self.max:
                self.max = self.w(P)
                self.res = P
        while len(C) != 0:
            if self.w(C) + self.w(P) <= self.max:
                return
            ic, i = self.min_order(C)
            if self.maxi[ic] + self.w(P) <= self.max:
                return
            C = C - {i}
            Pp = {i}.union(P)
            Cp = set()
            for v in C:
                if self.is_splex(Pp.union({v})):
                    Cp.add(v)
            self.find_max(Cp, Pp)


    def solve(self):
        self.order = list(np.argsort([self.G.nodes[i]["weight"] for i in self.G.nodes()]))
        print(len(self.order))
        for i in reversed(range(self.N)):
            print(i)
            C = set([self.order[x] for x in range(i + 1, self.N)])
            self.find_max(C, {self.order[i]})
            self.maxi[i] = self.max


class SplexRunner:
    def __init__(self, config):
        self.save_dir = config.save_dir
        self.dataset_config = config.dataset
        self.model_config = config.model
        self.train_config = config.train
        self.test_config = config.test
        self.output_columns = [
            'stocks', 'weights',
            'corr_min', 'corr_max', 'corr_mean', 'corr_std',
            'return', 'sigma', 'sharpe', 'information', 'modigliani',
        ]
    
    
    def train(self):
        train_dataset = eval(self.dataset_config.loader_name)(self.dataset_config, self.train_config)
        
        corrs = train_dataset.corr().fillna(1)
        
        asset_performances = train_dataset.tail(1).reset_index(drop=True) / train_dataset.head(1).reset_index(drop=True) - 1
        
        G = nx.Graph()
        
        for asset in asset_performances.columns:
            G.add_node(asset, weight=asset_performances.loc[0, asset])
        
        for i in corrs.index:
            for j in corrs.columns:
                i_j_corr = corrs.loc[i, j]
                if i_j_corr < self.model_config.weight_limit:
                    G.add_edge(i, j, weight=i_j_corr)
        
        solver = SplexSolver(G, self.model_config.s)
        solver.solve()
        print(solver.res)
    
    
    def test(self):
        pass