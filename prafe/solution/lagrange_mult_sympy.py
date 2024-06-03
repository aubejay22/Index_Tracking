from prafe.solution.solution import Solution
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import cvxpy as cp
import pandas as pd
import time
from sympy import symbols, Eq, solve

import numpy as np
from prafe.portfolio import Portfolio
from prafe.universe import Universe


class Solution(Solution):
    def __init__(
        self,
        universe: Universe,
        portfolio: Portfolio,
        solution_name: str,
        method: str,
        N=None,
        K=None,
    ):
        self.portfolio = portfolio
        self.universe = universe
        self.solution_name = solution_name
        self.method = method
        self.num_assets = N
        self.K = K
        
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        
        self.new_return = np.array(self.universe.df_return)
        self.new_index = np.array(self.universe.df_index)
        self.stock_list = self.universe.stock_list
    
    def objective_function(self, weight: list) -> list:
        error = self.new_return @ weight - self.new_index
        error = np.sum(error**2)
        return error
    
    def system_of_equations_sympy(self, vars):
        # SymPy variables
        num_lambdas = self.num_assets + 2
        weights = vars[:-num_lambdas]
        lambdas = vars[-num_lambdas:]
        
        coefficient = 1000
        eps = 1e-4
        
        # Define symbols for weights and lambdas
        weight_symbols = symbols(f'w0:{self.num_assets}')
        lambda_symbols = symbols(f'l0:{num_lambdas}')
        
        equations = []
        
        # Objective function derivative
        for i in range(self.num_assets):
            objective_derivative = 2 * (self.new_return[:, i] @ (self.new_return @ weights - self.new_index))
            constraint1_derivative = lambdas[i]
            constraint2_derivative = lambdas[i+1]
            constraint3_derivative = lambdas[i+2] * (-coefficient) * (np.exp(-(coefficient * (weights[i] - eps)))) / ((1 + np.exp(-(coefficient * (weights[i] - eps)))) ** 2)
            equations.append(objective_derivative + constraint1_derivative + constraint2_derivative + constraint3_derivative)
        
        # Constraint for lambda 1 to N
        for i in range(self.num_assets):
            equations.append(weights[i])
        
        # Constraint for lambda N+1 (sum of weights)
        equations.append(np.sum(weights) - 1)
        
        # Constraint for lambda N+2 (sum of sigmoid applied weights)
        count = 1 / (1 + np.exp(-(coefficient * (weights - eps))))
        equations.append(np.sum(count) - self.K)
        
        return equations
    
    def lagrange_partial_ours(self) -> dict:
        seed_value = 42
        np.random.seed(seed_value)
        coefficient = 1000
        eps = 1e-4
        num_lambdas = self.num_assets + 2
        
        print("Lagrange function optimization with SymPy")
        
        trial = 1
        while True:
            start_time = time.time()
            initial_variable = np.random.rand(self.num_assets)
            initial_variable /= initial_variable.sum()  
            
            lambda_guess = np.random.rand(num_lambdas)
            
            # Combine initial guesses
            initial_guess = np.concatenate([initial_variable, lambda_guess])
            
            # Define symbols for weights and lambdas
            weight_symbols = symbols(f'w0:{self.num_assets}')
            lambda_symbols = symbols(f'l0:{num_lambdas}')
            all_symbols = weight_symbols + lambda_symbols
            
            # Define the system of equations using SymPy
            equations = self.system_of_equations_sympy(initial_guess)
            sympy_eqs = [Eq(eq, 0) for eq in equations]
            
            # Solve the system of equations
            result = solve(sympy_eqs, all_symbols)
            
            if result:
                weights = [result[weight_symbols[i]] for i in range(self.num_assets)]
                print(f'Optimal solution: {weights}')
                print(f"Objective: {self.objective_function(weights)}")
                print(f"Weight sum: {np.sum(weights)}")
                
                topK_weight_sum = 0
                sorted_weights = sorted(weights, reverse=True)
                for weight in sorted_weights[:self.K]:
                    topK_weight_sum += weight
                print(f"Top K weight sum: {topK_weight_sum}")
                print(f"Cardinality: {np.sum(1 / (1 + np.exp(-(coefficient * (weights - eps)))))}")
                
                print(f"Inference time: {time.time() - start_time}")
                
                self.optimal_weight = weights
                self.stock2weight = {self.stock_list[i]: weights[i] for i in range(len(self.stock_list))}
                
                self.portfolio.update_portfolio(self.stock2weight)
                self.optimal_error = self.objective_function(self.optimal_weight)
                print(f"Calculated error: {self.optimal_error}")
                break
            
            trial += 1
        
        return self.stock2weight, self.optimal_error
    
    def update_portfolio(self) -> dict:
        solution_name = self.solution_name
        weights = self.lagrange_partial_ours()
        return weights