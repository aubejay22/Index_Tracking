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
        
        self.new_return = np.array(self.universe.df_return)
        self.new_index = np.array(self.universe.df_index)
        
        
        self.stock_list = self.universe.stock_list
        
        self.eps = 1e-4
        self.coefficient = 10000
        self.penalty_ratio = 50
        
        # print(self.new_index)
        # print(self.new_return)
        # raise Exception("Finish")
    
    #! ablation 
    def penalty(self, weight):
        eps = self.eps
        coefficient = self.coefficient

        weight = 1 / ( 1 + np.e ** ( - coefficient * ( weight - eps) ) ) # before 1e-4
        g1 = - np.sum(weight) + self.K
        penalty_value = 0
        if g1 < 0:
            penalty_value += g1**2  # 제약 조건 위반 시 벌점 부여
        return penalty_value
    
    
    def objective_function(
        self,
        weight : list,
    ) -> list :
        # 누적 수익률
        error = self.new_return @ weight - self.new_index
        error = np.sum(error**2)
        
        return error# + self.penalty_ratio * self.penalty(weight) #! ablation 
    
    
    def weight_sum_constraint(
        self,
        weight : list,
    ) -> list :
        # sorted_weights = sorted(weight)
        # return np.sum(sorted_weights[:self.K]) - 1
        
        return np.sum(weight) - 1
    


    def cardinality_constraint2(
        self,
        weight : list,
    ):
        eps = self.eps
        # Approximated extended cardinality constraint
        # 99999999
        coefficient = self.coefficient

        # print("weight sum :", np.sum(weight))
        # print("weight :", weight)
        weight = 1 / ( 1 + np.e ** ( - coefficient * ( weight - eps) ) ) # before 1e-4
        # print("binary :", weight)
        # print("num of stocks :", np.sum(weight))
        # print("min num of stocks :", self.K)
        return - np.sum(weight) + self.K 
    

    def lagrange_partial_ours(
        self
    ) -> dict :
        seed_value = 42
        np.random.seed(seed_value)
        
        trial = 1
        while(1):
            start_time = time.time()
            # Define initial weight
            initial_weight = np.random.rand(self.num_assets)
            initial_weight /= initial_weight.sum()  
            bounds = [(0, 1) for _ in range(self.num_assets)]

            # Define Constraints
            # constraint = [{'type': 'eq', 'fun': self.weight_sum_constraint},
            #               {'type': 'ineq', 'fun': self.cardinality_constraint}]
            #! ablation 
            constraint = [{'type': 'eq', 'fun': self.weight_sum_constraint},# 'jac': self.weight_sum_jac},
                        {'type': 'ineq', 'fun': self.cardinality_constraint2}]#, 'jac': self.cardinality_jac}]
            
            # Optimization
            print(len(initial_weight))
            result = minimize(self.objective_function, initial_weight, method = self.method, constraints=constraint, bounds=bounds, options={'maxiter': 200})#, tol=1e-6)
            self.optimal_weight = result.x
            self.stock2weight = {}
            for i in range(len(self.stock_list)):
                if result.x[i] <= 1e-4:
                    self.stock2weight[self.stock_list[i]] = 0
                else:
                    self.stock2weight[self.stock_list[i]] = result.x[i]
                
            # Update Portfolio & Calculate Error
            self.portfolio.update_portfolio(self.stock2weight)
            
            # error = self.new_return @ self.optimal_weight - self.new_index
            # error = np.sum(error**2)
            # ! ablation 
            self.optimal_error = self.objective_function(self.optimal_weight)
            # self.optimal_error = error
            print(f"Calculated error : {self.optimal_error}")
            
            # Calculate Top K weight sum
            topK_weight_sum = 0
            sorted_weights = sorted(self.stock2weight.items(), key=lambda x: x[1], reverse=True)
            for stock, weight in sorted_weights[:self.K]:
                topK_weight_sum += weight
            
            count = 0
            for stock, weight in sorted_weights:
                if (weight >= 0.0001):
                    count += 1
                
                
            # To avoid local optima
            if trial > 50:
                print("No portfolio satisfies the constraints")
                break
            # #! ablation 
            # if topK_weight_sum < 0.97 or topK_weight_sum > 1.01:
            #     print("topK weight sum", topK_weight_sum)
            #     trial += 1
            #     continue
            # if (count > self.K):
            #     print("count:", count)
            #     trial += 1
            #     continue
            w = 1 / ( 1 + np.e ** ( - self.coefficient * ( self.optimal_weight - self.eps) ) ) # before 1e-4
            print(f"calculated cardinality: {np.sum(w)}")
            print(f"K: {self.K}")
            print("Cardinality does not satisfied")
            # ! ablation 
            if self.cardinality_constraint2(self.optimal_weight) < 0:
                w = 1 / ( 1 + np.e ** ( - self.coefficient * ( self.optimal_weight - self.eps) ) ) # before 1e-4
                print(f"calculated cardinality: {np.sum(w)}")
                print(f"K: {self.K}")
                print("Cardinality does not satisfied")
                break
            else:
                print("trial : ", trial)
                print(f'sec : {time.time() - start_time}')
                print(f'min : {(time.time() - start_time)/60}')
                # print(result)
                break
            
        return self.stock2weight, self.optimal_error
    
    
    def update_portfolio(
        self,
    ) -> dict :
        solution_name = self.solution_name

        if solution_name == 'lagrange_full':
            weights = self.lagrange_full_replication()
        elif solution_name == 'lagrange_ours':
            weights = self.lagrange_partial_ours()
        elif solution_name == 'lagrange_ours2':
            weights = self.lagrange_partial_ours2()
        elif solution_name == 'lagrange_forward':
            weights = self.lagrange_partial_forward()
        elif solution_name == 'lagrange_backward':
            weights = self.lagrange_partial_backward()
        elif solution_name == 'QP_full':
            weights = self.QP_full_replication()
        elif solution_name == 'QP_forward':
            weights = self.QP_partial_forward()
        elif solution_name == 'QP_backward':
            weights = self.QP_partial_backward()
        
        return weights
        
