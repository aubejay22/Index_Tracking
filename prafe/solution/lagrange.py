import numpy as np
from prafe.portfolio import Portfolio
from prafe.universe import Universe
from prafe.objective import cumulative_return, variance, mdd, mdd_duration
from prafe.constraint.constraint import weights_sum_constraint, stocks_number_constraint

from prafe.solution.solution import Solution
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import cvxpy as cp
import numpy as np
import pandas as pd
from sympy import symbols, Eq, solve
from scipy.optimize import minimize

## Add your Strategy Here!

class Lagrange(Solution):
    
    
    def __init__(
        self,
        universe : Universe,
        portfolio : Portfolio,
        solution_name : str,
        k = None,
        ):
        self.stock_list = universe.stock_list
        self.portfolio = portfolio
        self.universe = universe
        
        self.new_return = np.array(self.universe.df_return)
        self.new_index = np.array(self.universe.df_index)
        self.num_assets = len(self.new_return[0])
        self.solution_name = solution_name
        self.K = k

    
    def objective_function(
        self,
        weight : list,
    ) -> list :
        error = self.new_return @ weight - self.new_index
        return np.sum(error**2)
    
    
    def weight_sum_constraint(
        self,
        weight : list,
    ) -> list :
        return np.sum(weight) - 1
    
    
    def cardinality_constraint(
        self,
        weight : list,
    ) -> list :
        eps = 1e-4
        coefficient = 99999999999
        approximated_num_of_stocks = [ 1 - 1 / ( coefficient * w + 1 ) for w in weight ]
        approximated_num_of_stocks = np.sum(approximated_num_of_stocks) 
        print("num of stocks :", approximated_num_of_stocks)
        print("max num of stocks :", self.K)
        return - approximated_num_of_stocks + self.K - 1  # (number of stocks) >= (min number of stocks)

    
    
    def stock_ratio_constraint(
        self,
        weight : list,
    ):
        eps = 1e-4
        min_number_of_stocks = self.strategy.min_stock_ratio[0]
        min_ratio = self.strategy.min_stock_ratio[1]
        # Approximated extended cardinality constraint
        coefficient = 100000000
        approximated_num_of_stocks = [ 1 / ( 1 + np.e ** ( - (coefficient * ( w - min_ratio )) ) ) for w in weight ]
        approximated_num_of_stocks = np.sum(approximated_num_of_stocks) 
        print("num of stocks :", approximated_num_of_stocks)
        print("min num of stocks :", min_number_of_stocks)
        return approximated_num_of_stocks - min_number_of_stocks + eps  # (number of stocks) >= (min number of stocks)
    
        
    def lagrange_full_replication(
        self
    ) -> dict :
        
        initial_weight = np.random.rand(self.num_assets)
        initial_weight /= initial_weight.sum()  
        bounds = [(0, 1) for _ in range(len(self.expected_return))]

        constraint = {'type': 'eq', 'fun': self.weight_sum_constraint}
        result = minimize(self.objective_function, initial_weight, method = 'SLSQP', constraints=constraint, bounds=bounds)
        self.optimal_weight_full = result.x

        self.stock2weight_full = {}
        for i in range(len(self.stock_list)):
            self.stock2weight_full[self.stock_list[i]] = result.x[i]
            
        self.portfolio.update_portfolio(self.stock2weight_full)
        self.optimal_error_full = self.objective_function(self.optimal_weight_full)

        print(f"Calculated error : {self.optimal_weight_full}")

        return self.stock2weight_full
    
    
    def lagrange_partial_ours(
        self
    ) -> dict :
        
        initial_weight = np.random.rand(self.num_assets)
        initial_weight /= initial_weight.sum()  
        bounds = [(0, 1) for _ in range(len(self.expected_return))]

        constraint = {'type': 'eq', 'fun': self.weight_sum_constraint,
                      'type': 'ineq', 'fun': self.cardinality_constraint}
        result = minimize(self.objective_function, initial_weight, method = 'SLSQP', constraints=constraint, bounds=bounds)
        self.optimal_weight_ours = result.x

        self.stock2weight_ours = {}
        for i in range(len(self.stock_list)):
            self.stock2weight_ours[self.stock_list[i]] = result.x[i]
            
        self.portfolio.update_portfolio(self.stock2weight_ours)
        self.optimal_error_ours = self.objective_function(self.optimal_weight_ours)

        print(f"Calculated error : {self.optimal_weight_ours}")

        return self.stock2weight_ours
    
    
    # def lagrange_partial_forward(
    #     self
    # ) -> dict :
    
    
    def update_portfolio(
        self,
    ) -> dict :
        solution_name = self.solution_name
        if solution_name == 'lagrange_full':
            weights = self.lagrange_full_replication()
        elif solution_name == 'lagrange_ours':
            weights = self.lagrange_partial_ours()
        
        return weights
        
