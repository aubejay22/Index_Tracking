import numpy as np
import cvxpy as cp
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns, objective_functions
from prafe.evaluation import Evaluator
from prafe.portfolio import Portfolio
from prafe.universe import Universe
from prafe.strategy import GenerativeStrategy
from prafe.objective import cumulative_return, variance, mdd, mdd_duration
from prafe.constraint.constraint import weights_sum_constraint, variance_constraint, mdd_constraint, mdd_duration_constraint, cumulative_return_constraint, stocks_number_constraint, industry_ratio_constraint, stock_ratio_constraint

from sklearn.model_selection import ParameterGrid, ParameterSampler
import time
import random
import torch
import matplotlib.pyplot as plt
from pykeops.torch import LazyTensor
from kneed import KneeLocator

use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
device_id = "cuda:1" if use_cuda else "cpu"

eps = 1e-6

class Solution():

    def __init__(
        self,
        universe : Universe,
        portfolio : Portfolio,
        strategy: GenerativeStrategy
        ):
        self.stock_list = universe.stock_list
        self.portfolio = portfolio
        self.universe = universe
        self.strategy = strategy

        self.objective = strategy.objective
        self.initial_weights = portfolio.get_weights()

    def compute_objective(
        self
    ) -> float :
        
        if self.objective == "cumulative_return":
            return cumulative_return(self.portfolio, self.universe)
        elif self.objective == "variance":
            return variance(self.portfolio, self.universe)
        elif self.objective == "mdd":
            return mdd(self.portfolio, self.universe)
        elif self.objective == "mdd_duration":
            return mdd_duration(self.portfolio, self.universe)
        else:
            raise NotImplementedError

    def update_portfolio(
        self
    ) -> dict :
        
        # TODO: Implement the update portfolio function
        weights = {}
        for stock_code in self.stock_list:
            weights[stock_code] = 0.0
        
        self.portfolio.update_portfolio(weights)
        print("Portfolio updated")
        print(f"Calculated objective: {self.compute_objective()}")

        return weights
    
    def update_rewards(
        self
    ) -> np.ndarray :
        
        rewards = np.array([])
        self.portfolio.rewards = rewards
        
        return self.portfolio.get_rewards()
    
    def _does_satisfy_constraints(self) -> bool:
        if self.strategy.max_weights_sum is not None and weights_sum_constraint(self.portfolio, self.universe, self.strategy) == False:
            return False
        if self.strategy.max_variance is not None and variance_constraint(self.portfolio, self.universe, self.strategy) == False:
            return False
        if self.strategy.max_mdd is not None and mdd_constraint(self.portfolio, self.universe, self.strategy) == False:
            return False
        if self.strategy.max_mdd_duration is not None and mdd_duration_constraint(self.portfolio, self.universe, self.strategy) == False:
            return False
        if self.strategy.min_cumulative_return is not None and cumulative_return_constraint(self.portfolio, self.universe, self.strategy) == False:
            return False
        if self.strategy.max_stocks_number_constraint is not None and stocks_number_constraint(self.portfolio, self.universe, self.strategy) == False:
            return False
        if self.strategy.min_industry_ratio is not None and industry_ratio_constraint(self.portfolio, self.universe, self.strategy) == False:
            return False
        if self.strategy.min_stock_ratio is not None and stock_ratio_constraint(self.portfolio, self.universe, self.strategy) == False:
            return False
        return True