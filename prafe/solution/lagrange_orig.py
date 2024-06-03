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
    
    
    # def weight_sum_jac(
    #     self,
    #     weight : list,
    # ) -> list :
    #     # print(np.ones)
    #     return np.ones(len(weight))
    
    
    def cardinality_constraint(
        self,
        weight : list,
    ) -> list :
        eps = 1e-4
        coefficient = 99999999999
        weight = 1 - 1 / ( coefficient * weight + 1 )
        # print("num of stocks :", np.sum(weight))
        # print("max num of stocks :", self.K)
        return - np.sum(weight) + self.K - eps  # (number of stocks) >= (min number of stocks)
    
    
    # def cardinality_jac(
    #     self,
    #     weight : list,
    # ) -> list :
    #     coefficient = 99999999999
    #     # print(coefficient/((coefficient*weight+1)**2))
    #     return coefficient/((coefficient*weight+1)**2)

    
    
    def cardinality_constraint2(
        self,
        weight : list,
    ):
        eps = 1e-4
        # Approximated extended cardinality constraint
        coefficient = 99999999999

        weight = 1 / ( 1 + np.e ** ( - coefficient * ( weight -1e-4) ) )
        # print("num of stocks :", np.sum(weight))
        # print("min num of stocks :", self.K)
        return - np.sum(weight) + self.K - eps 
    
    

    def lagrange_full_replication(
        self
    ) -> dict :
        # Define initial weight
        initial_weight = np.ones(self.num_assets)
        initial_weight /= initial_weight.sum()  
        bounds = [(0, 1) for _ in range(self.num_assets)]

        # Define Constraints    
        constraint = {'type': 'eq', 'fun': self.weight_sum_constraint}#, 'jac': self.weight_sum_jac}
        # constraint = {'type': 'eq', 'fun': self.weight_sum_constraint}
        
        # Optimization
        result = minimize(self.objective_function, initial_weight, method = self.method, constraints=constraint, bounds=bounds)
        self.optimal_weight = result.x
        self.stock2weight = {}
        for i in range(len(self.stock_list)):
            self.stock2weight[self.stock_list[i]] = result.x[i]
        
        # Update Portfolio & Calculate Error
        self.portfolio.update_portfolio(self.stock2weight)
        self.optimal_error = self.objective_function(self.optimal_weight)
        print(f"Calculated error : {self.optimal_error}")

        # print(result)
        return self.stock2weight, self.optimal_error
    
    
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
            constraint = [{'type': 'eq', 'fun': self.weight_sum_constraint},# 'jac': self.weight_sum_jac},
                        {'type': 'ineq', 'fun': self.cardinality_constraint2}]#, 'jac': self.cardinality_jac}]
            
            # Optimization
            result = minimize(self.objective_function, initial_weight, method = self.method, constraints=constraint, bounds=bounds, tol=1e-6)
            self.optimal_weight = result.x
            self.stock2weight = {}
            for i in range(len(self.stock_list)):
                self.stock2weight[self.stock_list[i]] = result.x[i]
                
            # Update Portfolio & Calculate Error
            self.portfolio.update_portfolio(self.stock2weight)
            self.optimal_error = self.objective_function(self.optimal_weight)
            print(f"Calculated error : {self.optimal_error}")
            
            # Calculate Top K weight sum
            topK_weight_sum = 0
            sorted_weights = sorted(self.stock2weight.items(), key=lambda x: x[1], reverse=True)
            for stock, weight in sorted_weights[:self.K]:
                topK_weight_sum += weight
                
            # To avoid local optima
            if trial > 50:
                print("No portfolio satisfies the constraints")
                break
            if topK_weight_sum < 0.96 or topK_weight_sum > 1.01:
                print(topK_weight_sum)
                trial += 1
                continue
            else:
                print("trial : ", trial)
                print(f'sec : {time.time() - start_time}')
                print(f'min : {(time.time() - start_time)/60}')
                # print(result)
                break
            
        return self.stock2weight, self.optimal_error
    
    
    def lagrange_partial_forward(
        self
    ) -> dict :
        num_assets = self.num_assets
        new_return = self.new_return
        new_index = self.new_index
        largest_weight = []
        largest_stocks = []
        stock_list = self.stock_list
        K = self.K
        
        while len(largest_weight) < K :  
            # Define initial weight 
            initial_weight = np.ones(num_assets)
            initial_weight /= initial_weight.sum()  
            bounds = [(0, 1) for _ in range(num_assets)]
            
            # Define Objective & Constratins 
            objective = lambda weight: np.sum((new_return @ weight - new_index)**2)
            constraint = {'type': 'eq', 'fun': self.weight_sum_constraint} #'jac': self.weight_sum_jac}
            # constraint = {'type': 'eq', 'fun': self.weight_sum_constraint}
        
            # Optimization
            result = minimize(objective, initial_weight, method = self.method, constraints=constraint, bounds=bounds)
            
            # Find Largest Weight
            max_idx = np.argmax(result.x)
            max_weight = result.x[max_idx]
            max_weight_stock = stock_list[max_idx]
            largest_weight.append(max_weight)
            largest_stocks.append(max_weight_stock)
            print("largest weight:", max_weight)
            print("largest weight stock:", max_weight_stock)
            
            # Remove Largest Weight
            new_return = np.delete(new_return, max_idx, axis=1)
            stock_list = np.delete(stock_list, max_idx)
            num_assets -= 1
            
        # Finally QP with K stocks
        initial_weight = np.ones(K)
        initial_weight /= initial_weight.sum()  
        bounds = [(0, 1) for _ in range(K)]
        # Define Largest Return data
        df_new_return = self.universe.df_return[largest_stocks]
        new_return = np.array(df_new_return)
        # Define Objective & Constratins & Problem
        objective = lambda weight: np.sum((new_return @ weight - new_index)**2)
        constraint = {'type': 'eq', 'fun': self.weight_sum_constraint}#, 'jac': self.weight_sum_jac}
        # Optimization
        result = minimize(objective, initial_weight, method = self.method, constraints=constraint, bounds=bounds)
        self.optimal_weight = result.x
        self.stock2weight = {}
        for i in range(len(largest_stocks)):
            self.stock2weight[largest_stocks[i]] = result.x[i]
        for i in range(len(stock_list)):
            if stock_list[i] not in self.stock2weight:
                self.stock2weight[stock_list[i]] = 0
        
        # Update Portfolio & Calculate Error
        self.portfolio.update_portfolio(self.stock2weight)
        self.optimal_error = sum((new_return @ self.optimal_weight - new_index)**2)
        print(f"Calculated error : {self.optimal_error}")

        return self.stock2weight, self.optimal_error
    
    
    def lagrange_partial_backward(
        self,
    ) -> dict :
        num_assets = self.num_assets
        new_return = self.new_return
        new_index = self.new_index
        smallest_weight = []
        smallest_stocks = []
        stock_list = self.stock_list
        K = self.K
        
        new_stock_list = stock_list
        while num_assets >= K :  
            # Define initial weight 
            initial_weight = np.ones(num_assets)
            initial_weight /= initial_weight.sum()  
            bounds = [(0, 1) for _ in range(num_assets)]
            
            # Define Objective & Constratins 
            objective = lambda weight: np.sum((new_return @ weight - new_index)**2)
            constraint = {'type': 'eq', 'fun': self.weight_sum_constraint}#, 'jac': self.weight_sum_jac}
            # constraint = {'type': 'eq', 'fun': self.weight_sum_constraint}
        
            # Optimization
            result = minimize(objective, initial_weight, method = self.method, constraints=constraint, bounds=bounds)
            
            if num_assets != K:
                # Find Smallest Weight
                min_idx = np.argmin(result.x)
                min_weight = result.x[min_idx]
                min_weight_stock = new_stock_list[min_idx]
                smallest_weight.append(min_weight)
                smallest_stocks.append(min_weight_stock)
                print("smallest weight:", min_weight)
                print("smallest weight stock:", min_weight_stock)
                
                # Remove Smallest Weight
                new_return = np.delete(new_return, min_idx, axis=1)
                new_stock_list = np.delete(new_stock_list, min_idx)
                num_assets -= 1
            else:
                break
        
        self.optimal_weight = result.x
        self.stock2weight = {}
        for i in range(len(new_stock_list)):
            self.stock2weight[new_stock_list[i]] = self.optimal_weight[i]
        for i in range(len(stock_list)):
            if stock_list[i] not in self.stock2weight:
                self.stock2weight[stock_list[i]] = 0
        
        # Update Portfolio & Calculate Error
        self.portfolio.update_portfolio(self.stock2weight)
        self.optimal_error = sum((new_return @ self.optimal_weight - new_index)**2)
        print(f"Calculated error : {self.optimal_error}")
        return self.stock2weight, self.optimal_error
        
        
    
    def QP_full_replication(
        self
    ) -> dict :
        # Define initial weight & Error
        initial_weight = cp.Variable(self.num_assets)
        error = self.new_return @ initial_weight - self.new_index
        
        # Define Objective & Constratins & Problem
        objective = cp.Minimize(cp.sum_squares(error))
        constraint = [cp.sum(initial_weight) == 1, initial_weight >= 0]
        problem = cp.Problem(objective, constraint)
        
        # Optimization
        problem.solve(solver='OSQP',verbose=False)
        self.optimal_weight = initial_weight.value
        self.stock2weight = {}
        for i in range(len(self.stock_list)):
            self.stock2weight[self.stock_list[i]] = self.optimal_weight[i]
        
        # Update Portfolio & Calculate Error
        self.portfolio.update_portfolio(self.stock2weight)
        self.optimal_error = self.objective_function(self.optimal_weight)
        print(f"Calculated error : {self.optimal_error}")

        return self.stock2weight, self.optimal_error
    
    
    def QP_partial_forward(
        self
    ) -> dict :
        num_assets = self.num_assets
        new_return = self.new_return
        new_index = self.new_index
        largest_weight = []
        largest_stocks = []
        stock_list = self.stock_list
        K = self.K
        
        while len(largest_weight) < K :  
            # Define initial weight & Error
            initial_weight = cp.Variable(num_assets)
            error = new_return @ initial_weight - new_index
            
            # Define Objective & Constratins & Problem
            objective = cp.Minimize(cp.sum_squares(error))
            constraint = [cp.sum(initial_weight) == 1, initial_weight >= 0]
            problem = cp.Problem(objective, constraint)
            
            # Optimization
            problem.solve(solver='OSQP',verbose=False)
            
            # Find Largest Weight
            max_idx = np.argmax(initial_weight.value)
            max_weight = initial_weight.value[max_idx]
            max_weight_stock = stock_list[max_idx]
            largest_weight.append(max_weight)
            largest_stocks.append(max_weight_stock)
            print("largest weight:", max_weight)
            print("largest weight stock:", max_weight_stock)
            
            # Remove Largest Weight
            new_return = np.delete(new_return, max_idx, axis=1)
            stock_list = np.delete(stock_list, max_idx)
            num_assets -= 1
            
        # Finally QP with K stocks
        initial_weight = cp.Variable(K)
        # Define Largest Return data
        df_new_return = self.universe.df_return[largest_stocks]
        new_return = np.array(df_new_return)
        error = new_return @ initial_weight - new_index
        # Define Objective & Constratins & Problem
        objective = cp.Minimize(cp.sum_squares(error))
        constraints = [cp.sum(initial_weight) == 1, initial_weight >= 0]
        problem = cp.Problem(objective, constraints)
        # Optimization
        problem.solve(solver='OSQP',verbose=False)
        self.optimal_weight = initial_weight.value
        self.stock2weight = {}
        for i in range(len(largest_stocks)):
            self.stock2weight[largest_stocks[i]] = self.optimal_weight[i]
        for i in range(len(stock_list)):
            if stock_list[i] not in self.stock2weight:
                self.stock2weight[stock_list[i]] = 0
        
        # Update Portfolio & Calculate Error
        self.portfolio.update_portfolio(self.stock2weight)
        self.optimal_error = sum((new_return @ self.optimal_weight - new_index)**2)
        print(f"Calculated error : {self.optimal_error}")

        return self.stock2weight, self.optimal_error
    
    
    def QP_partial_backward(
        self,
    ) -> dict :
        num_assets = self.num_assets
        new_return = self.new_return
        new_index = self.new_index
        smallest_weight = []
        smallest_stocks = []
        stock_list = self.stock_list
        K = self.K
        
        new_stock_list = stock_list
        while num_assets >= K :  
            # Define initial weight 
            initial_weight = cp.Variable(num_assets)
            error = new_return @ initial_weight - new_index
            
            # Define Objective & Constratins & Problem
            objective = cp.Minimize(cp.sum_squares(error))
            constraint = [cp.sum(initial_weight) == 1, initial_weight >= 0]
            problem = cp.Problem(objective, constraint)
        
            # Optimization
            problem.solve(solver='OSQP',verbose=False)
            
            if num_assets != K:
                # Find Smallest Weight
                min_idx = np.argmin(initial_weight.value)
                min_weight = initial_weight.value[min_idx]
                min_weight_stock = new_stock_list[min_idx]
                smallest_weight.append(min_weight)
                smallest_stocks.append(min_weight_stock)
                print("smallest weight:", min_weight)
                print("smallest weight stock:", min_weight_stock)
                
                # Remove Smallest Weight
                new_return = np.delete(new_return, min_idx, axis=1)
                new_stock_list = np.delete(new_stock_list, min_idx)
                num_assets -= 1
            else:
                break
        
        self.optimal_weight = initial_weight.value
        self.stock2weight = {}
        for i in range(len(new_stock_list)):
            self.stock2weight[new_stock_list[i]] = self.optimal_weight[i]
        for i in range(len(stock_list)):
            if stock_list[i] not in self.stock2weight:
                self.stock2weight[stock_list[i]] = 0
        
        # Update Portfolio & Calculate Error
        self.portfolio.update_portfolio(self.stock2weight)
        self.optimal_error = sum((new_return @ self.optimal_weight - new_index)**2)
        print(f"Calculated error : {self.optimal_error}")
        return self.stock2weight, self.optimal_error
    
    
    def update_portfolio(
        self,
    ) -> dict :
        solution_name = self.solution_name

        if solution_name == 'lagrange_full':
            weights = self.lagrange_full_replication()
        elif solution_name == 'lagrange_ours':
            weights = self.lagrange_partial_ours()
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
        
