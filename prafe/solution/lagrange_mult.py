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
import time
from sympy import symbols, Eq, solve
from scipy.optimize import minimize
from scipy.optimize import fsolve
from scipy.optimize import root

## Add your Strategy Here!

class Solution(Solution):
    
    
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
        
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)  
        
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
        
        error = self.new_return @ weight - self.new_index
        error = np.sum(error**2)
        
        return error 
    
    
    def derivative_weights(
        self,
        weight,
        lambdas,
        i
    ):
        coefficient = 1000
        eps = 1e-4
        return 2 * (self.new_return[:,i] @ (self.new_return @ weight - self.new_index)) + lambdas[0]#+ lambdas[i] + lambdas[self.num_assets]# + lambdas[1] * coefficient * (np.e ** (-(coefficient * (weight[i] - eps)))) / ((1 + np.e ** (-(coefficient * (weight[i] - eps)))) ** 2)
    
    
    def derivative_lambda1(
        self,
        weight,
        lambdas,
    ):
        return np.sum(weight) - 1
    
    
    def derivative_lambda2(
        self,
        weight,
        lambdas,
    ):
        coefficient = 1000
        eps = 1e-4
        return np.sum(1 / (1 + np.e ** (-(coefficient * (weight - eps)))) ) - self.K
    
    
    def system_of_equations_full(
        self,
        vars
    ):
        num_lambdas = 1
        weight = vars[:-num_lambdas]
        lambdas = vars[-num_lambdas:]
        coefficient = 1000
        eps = 1e-4
        
        equations = []
        
        # derivative for weights
        for i in range(self.num_assets):
            objective_der = 2 * (self.new_return[:,i] @ (self.new_return @ weight - self.new_index))
            const1_der = lambdas[0]
            equations.append(objective_der + const1_der)
        # derivative for lambda1
        equations.append(np.sum(weight) - 1)
        
        return equations
    
    # new version~
    def system_of_equations_version1(
        self,
        vars
    ):
        num_lambdas = self.num_assets + 2
        # print(num_lambdas)
        weight = vars[:-num_lambdas]
        lambdas = vars[-num_lambdas:]
        coefficient = 1000
        eps = 1e-4
        
        # print(self.new_return[:,0].shape)
        # print(self.new_return.shape)
        # print(weight.shape)
        # print(self.new_index.shape)
        
        equations = []
        # derivative for weights
        for i in range(self.num_assets):
            objective_der = 2 * (self.new_return[:,i] @ (self.new_return @ weight - self.new_index))
            const1_der = lambdas[i]
            const2_der = lambdas[i+1]
            const3_der = lambdas[i+2] * (- coefficient) * (np.e ** (-(coefficient * (weight[i] - eps)))) / ((1 + np.e ** (-(coefficient * (weight[i] - eps)))) ** 2)
            equations.append(objective_der + const1_der + const2_der + const3_der)
        
            # derivative for lambda 1 to N
            equations.append(weight[i])
            
        # derivative for lambda N+1
        equations.append(np.sum(weight) - 1)
        
        # derivative for lambda N+2
        count = 1 / (1 + np.e ** (-(coefficient * (weight - eps)))) 
        equations.append(np.sum(count) - self.K)
        
        return equations
    
    
    def system_of_equations_version3(
        self,
        vars
    ):
        num_lambdas = 2
        weight = vars[:-num_lambdas]
        lambdas = vars[-num_lambdas:]
        coefficient = 1000
        eps = 1e-4
        
        equations = []
        
        # derivative for weights
        for i in range(self.num_assets):
            objective_der = 4 * (self.new_return[:,i] @ (self.new_return @ (weight ** 2) - self.new_index))
            const1_der = 2 * lambdas[0] * weight[i]
            const2_der = lambdas[1] * coefficient * weight[i] * (np.e ** (-(coefficient * (weight[i]**2 - eps)))) / ((1 + np.e ** (-(coefficient * (weight[i]**2 - eps)))) ** 2)
            equations.append(objective_der + const1_der + const2_der)
        # derivative for lambda1
        equations.append(np.sum(weight**2) - 1)
        
        # derivative for lambda2
        count = 1 / (1 + np.e ** (-(coefficient * (weight**2 - eps)))) 
        equations.append(np.sum(count) - self.K)
        return equations


    
    def lagrange_partial_ours(
        self
    ) -> dict :
        seed_value = 42
        np.random.seed(seed_value)
        coefficient = 1000
        eps = 1e-4
        num_lambdas = self.num_assets + 2
        
        print("lagrange function")
        
        # print(self.universe.df_return.shape)
        # print(self.new_return[:,0].shape)
        
        trial = 1
        while(1):
            start_time = time.time()
            # Define initial weight
            initial_variable = np.random.rand(self.num_assets)
            initial_variable /= initial_variable.sum()  
            # print(initial_variable)
            
            lambda_guess = np.random.rand(num_lambdas)
            slack_guess = np.random.rand(1)
            
            # Optimization
            result= root(self.system_of_equations_version1, np.concatenate([initial_variable, lambda_guess]), method='lm')#, full_output=True)#, slack_guess]))
            
            print(f'Success: {result.success}')
            print(f'optimal solution: {result.x}')
            print(f"objective: {self.objective_function(result.x[:-num_lambdas])}")
            print(f"weight sum: {np.sum(result.x[:-num_lambdas])}")
            topK_weight_sum = 0
            sorted_weights = sorted(result.x[:-num_lambdas],reverse=True)
            for weight in sorted_weights[:self.K]:
                topK_weight_sum += weight
            print(f"top K weight sum: {topK_weight_sum}")
            print(f"cardinality: {np.sum(1 / (1 + np.e ** (-(coefficient * (result.x[:-num_lambdas] - eps)))))}")
            
            # print(f"initial weight: {initial_variable}")
            # print(f'Success: {result[-2:]}')
            # print(f'optimal solution: {result[0]}')
            # print(f"objective: {self.objective_function(result[0][:-2])}")
            # print(f"weight sum: {np.sum(result[0][:-2])}")
            # topK_weight_sum = 0
            # sorted_weights = sorted(result[0][:-2],reverse=True)
            # for weight in sorted_weights[:self.K]:
            #     topK_weight_sum += weight
            # print(f"top K weight sum: {topK_weight_sum}")
            # print(f"cardinality: {np.sum(1 / (1 + np.e ** (-(coefficient * (result[0][:-2] - eps)))))}")
            print(f"inference time: {time.time() - start_time}")
            
            self.optimal_weight = result.x[:-num_lambdas]
            self.stock2weight = {}
            for i in range(len(self.stock_list)):
                self.stock2weight[self.stock_list[i]] = result.x[i]
                
            # Update Portfolio & Calculate Error
            self.portfolio.update_portfolio(self.stock2weight)
            self.optimal_error = self.objective_function(self.optimal_weight)
            print(f"Calculated error : {self.optimal_error}")
            break
            # raise Exception("")
            
        return self.stock2weight, self.optimal_error
    


    
    def update_portfolio(
        self,
    ) -> dict :
        solution_name = self.solution_name

        weights = self.lagrange_partial_ours()
    
        return weights
        
