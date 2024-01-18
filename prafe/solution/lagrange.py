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
        method : str,
        N = None,
        K = None,
        ):
        self.portfolio = portfolio
        self.universe = universe
        self.solution_name = solution_name
        self.method = method
        self.num_assets = N
        self.K = K
        
        self.new_return = np.array(self.universe.df_return)
        self.new_index = np.array(self.universe.df_index)
        
        self.stock_list = self.universe.stock_list
        
        # print(self.new_index)
        # print(self.new_return)
        # raise Exception("Finish")
    
    def objective_function(
        self,
        weight : list,
    ) -> list :
        # print(self.new_return @ weight)
        # print(self.new_index)
        # raise Exception("")
        error = self.new_return @ weight - self.new_index
        error = np.sum(error**2)
        # print(error)
        # raise Exception("")
        
        return error
    
    
    def weight_sum_constraint(
        self,
        weight : list,
    ) -> list :
        return np.sum(weight) - 1
    
    
    def weight_sum_jac(
        self,
        weight : list,
    ) -> list :
        # print(np.ones)
        return np.ones(len(weight))
    
    
    def cardinality_constraint(
        self,
        weight : list,
    ) -> list :
        eps = 1e-4
        coefficient = 99999999999
        approximated_num_of_stocks = [ 1 - 1 / ( coefficient * w + 1 ) for w in weight ]
        approximated_num_of_stocks = np.sum(approximated_num_of_stocks) 
        # print("num of stocks :", approximated_num_of_stocks)
        # print("max num of stocks :", self.K)
        return - approximated_num_of_stocks + self.K - eps  # (number of stocks) >= (min number of stocks)
    
    
    def cardinality_jac(
        self,
        weight : list,
    ) -> list :
        coefficient = 99999999999
        # print(coefficient/((coefficient*weight+1)**2))
        return coefficient/((coefficient*weight+1)**2)

    
    
    def cardinality_constraint2(
        self,
        weight : list,
    ):
        eps = 1e-4
        # Approximated extended cardinality constraint
        coefficient = 99999999999
        # approximated_num_of_stocks = [ 1 / ( 1 + np.e ** ( - (coefficient * ( w - eps )) ) ) for w in weight ]
        
        # approximated_num_of_stocks = 1 / ( 1 + np.e ** ( - (coefficient * ( weight - eps )) ) )
        # approximated_num_of_stocks = np.sum(approximated_num_of_stocks) 
        
        weight = 1 / ( 1 + np.e ** ( - (coefficient * ( weight - eps )) ) )
        return - np.sum(weight) + self.K - eps 
        # print("num of stocks :", approximated_num_of_stocks)
        # print("min num of stocks :", min_number_of_stocks)
        # return - approximated_num_of_stocks + self.K - eps  # (number of stocks) >= (min number of stocks)
    
        
    def lagrange_full_replication(
        self
    ) -> dict :
        
        initial_weight = np.ones(self.num_assets)
        initial_weight /= initial_weight.sum()  
        bounds = [(0, 1) for _ in range(self.num_assets)]

        constraint = {'type': 'eq', 'fun': self.weight_sum_constraint, 'jac': self.weight_sum_jac}
        # constraint = {'type': 'eq', 'fun': self.weight_sum_constraint}
        result = minimize(self.objective_function, initial_weight, method = self.method, constraints=constraint, bounds=bounds)
        self.optimal_weight_full = result.x

        self.stock2weight_full = {}
        for i in range(len(self.stock_list)):
            self.stock2weight_full[self.stock_list[i]] = result.x[i]
            
        self.portfolio.update_portfolio(self.stock2weight_full)
        self.optimal_error_full = self.objective_function(self.optimal_weight_full)

        print(f"Calculated error : {self.optimal_error_full}")

        return self.stock2weight_full
    
    
    def lagrange_partial_ours(
        self
    ) -> dict :
        
        trial = 1
        while(1):
            initial_weight = np.random.rand(self.num_assets)
            initial_weight /= initial_weight.sum()  
            bounds = [(0, 1) for _ in range(self.num_assets)]

            # constraint = [{'type': 'eq', 'fun': self.weight_sum_constraint},
            #               {'type': 'ineq', 'fun': self.cardinality_constraint}]
            constraint = [{'type': 'eq', 'fun': self.weight_sum_constraint, 'jac': self.weight_sum_jac},
                        {'type': 'ineq', 'fun': self.cardinality_constraint2}]#, 'jac': self.cardinality_jac}]
            result = minimize(self.objective_function, initial_weight, method = self.method, constraints=constraint, bounds=bounds)
            self.optimal_weight_ours = result.x

            self.stock2weight_ours = {}
            for i in range(len(self.stock_list)):
                self.stock2weight_ours[self.stock_list[i]] = result.x[i]
                
            self.portfolio.update_portfolio(self.stock2weight_ours)
            self.optimal_error_ours = self.objective_function(self.optimal_weight_ours)

            print(f"Calculated error : {self.optimal_error_ours}")
            
            topK_weight_sum = 0
            sorted_weights = sorted(self.stock2weight_ours.items(), key=lambda x: x[1], reverse=True)
            for stock, weight in sorted_weights[:self.K]:
                topK_weight_sum += weight
            # To avoid local optima
            if trial > 50:
                print("No portfolio satisfies the constraints")
                break
            if topK_weight_sum < 0.96:
                print(topK_weight_sum)
                trial += 1
                continue
            else:
                print("trial : ", trial)
                break

        

        return self.stock2weight_ours
    
    
    # def lagrange_partial_forward(
    #     self
    # ) -> dict :
    
    
    def QP_full_replication(
        self
    ) -> dict :
        
        initial_weight = cp.Variable(self.num_assets)
        error = self.new_return @ initial_weight - self.new_index
        
        objective = cp.Minimize(cp.sum_squares(error))
        constraint = [sum(initial_weight) == 1, initial_weight >= 0]
        
        problem = cp.Problem(objective, constraint)
        problem.solve(solver='OSQP',verbose=True)

        self.optimal_weight_full = initial_weight.value

        self.stock2weight_full = {}
        for i in range(len(self.stock_list)):
            self.stock2weight_full[self.stock_list[i]] = self.optimal_weight_full[i]
            
        self.portfolio.update_portfolio(self.stock2weight_full)
        self.optimal_error_full = cp.sum_squares(error)

        print(f"Calculated error : {self.optimal_error_full}")

        return self.stock2weight_full
    
    
    def update_portfolio(
        self,
    ) -> dict :
        solution_name = self.solution_name

        if solution_name == 'lagrange_full':
            weights = self.lagrange_full_replication()
        elif solution_name == 'lagrange_ours':
            weights = self.lagrange_partial_ours()
        elif solution_name == 'QP_full':
            weights = self.QP_full_replication()
        
        return weights
        
