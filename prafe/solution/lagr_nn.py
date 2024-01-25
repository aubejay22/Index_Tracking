import os
from torch import nn
from torch.nn import MultiheadAttention
from torch.nn.init import kaiming_normal_
from torch.nn.functional import one_hot, gumbel_softmax
import math 
import cvxpy as cp
from enum import IntEnum
import numpy as np

from tqdm import tqdm
from sklearn import metrics 
from solution import Solution


class Lagrange_Neural_Net(nn.Module, Solution):

    def __init__(self, universe, portfolio, solution_name, method, N, K):
        '''
            universe: new_universe
            portfolio: new_portfolio
            solution_name: args.solution_name, 
            method: args.method
            N: len(index_stocks_list)
            K: K
        '''
        
        self.universe = universe
        self.portfolio = portfolio
        self.solution_name = solution_name
        self.method = method
        self.num_assets = N
        self.K = K
        self.new_return = np.array(self.universe.df_return)
        self.new_index = np.array(self.universe.df_index)
        
        self.initial_weight = kaiming_normal_(cp.Variable(self.num_assets)) # for ReLU
        
        nn.Module.__init__(Lagrange_Neural_Net, self)
        Solution.__init__(self, self.universe, self.portfolio)

        # build a neural network
        self.emb = nn.Embedding(self.num_assets, K)
        self.ann = nn.Sequential()
        self.ann.add_module("theta1", nn.Linear(self.K, self.num_assets))
        self.ann.add_module("theta1_act", nn.ReLU())
        self.ann.add_module("theta2", nn.Linear(self.num_assets, self.K))
        self.ann.add_module("theta2_act", nn.ReLU())
        self.ann.add_module("theta3", nn.Linear(self.K, self.num_assets))
    
    
    def forward(self, new_return):
        
        x = new_return
        x = self.emb(x)
        
        new_weight = self.ann(x)

        
        return new_weight
    
    
    # in practice, there's a training session in here via main code. 
    def update_portfolio(self) -> dict:
        
        # train dataset
        train_input = self.new_index
        train_target = self.new_index
        
        # test dataset 
        test_input = self.new_index 
        test_target = self.new_index
        
        weight = self.initial_weight
        
        
        return weight