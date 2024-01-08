import numpy as np
from prafe.portfolio import Portfolio
from prafe.universe import Universe
from prafe.strategy import GenerativeStrategy
from prafe.objective import cumulative_return, variance, mdd, mdd_duration
from prafe.constraint.constraint import weights_sum_constraint, variance_constraint, mdd_constraint, mdd_duration_constraint, cumulative_return_constraint, stocks_number_constraint, industry_ratio_constraint, stock_ratio_constraint

from prafe.solution import Solution
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
        strategy: GenerativeStrategy
        ):
        self.stock_list = universe.stock_list
        self.portfolio = portfolio
        self.universe = universe
        self.strategy = strategy

        self.objective = strategy.objective
        self.initial_weights = portfolio.get_weights()

        
    
    def objectivefn(
        self,
        weight : list,
    ) -> list :
        objective_type = self.objective
        if objective_type == 'cumulative_return':   
            return - np.dot(weight, self.expected_return)
        elif objective_type == 'variance':
            variance = weight @ self.covariance_matrix @ weight
            return variance
        elif objective_type == 'mdd':
            MDD = mdd(self.portfolio, self.universe)
            # print(type(MDD))
            return MDD
        elif objective_type == 'mdd_duration':
            duration = mdd_duration(self.portfolio, self.universe)
            return duration
            
        # elif objective_type == 'mdd':
        # elif objective_type == 'mdd_duration': d
    
    
    def weight_sum_constraint(
        self,
        weight : list,
    ) -> list :
        max_weights_sum = self.strategy.max_weights_sum
        return np.sum(weight) - max_weights_sum  # (weight sum) == (max_weight_sum)
    
    
    def variance_constraint(
        self,
        weight : list,
    ) -> list :
        max_variance = self.strategy.max_variance
        variance = weight @ self.covariance_matrix @ weight
        return - variance + max_variance  # (variance) <= (max_variance)
    
        
    def return_constraint(
        self,
        weight : list,
    ) -> list :
        min_cumulative_return = self.strategy.min_cumulative_return
        weighted_return = np.dot(weight, self.expected_return) 
        return weighted_return - min_cumulative_return  # (return) >= (min_cumulative_return)
    
    
    def mdd_constraint(
        self,
        weight : list,
    ):
        max_mdd = self.strategy.max_mdd
        investments = self.portfolio.investments
        codes = list(investments.keys())
        weights = weight
        weights = np.array(weights)
        
        # Step 1: Compute the daily portfolio value
        portfolio_value = (self.universe.df_price[codes] * weights).sum(axis=1) 
        
        # Step 2: Compute the running maximum
        running_max = np.maximum.accumulate(portfolio_value)

        # Step 3: Compute the daily drawdown
        drawdown = portfolio_value / running_max - 1.0

        # Step 4: Compute the maximum drawdown
        MDD = drawdown.min()
        abs_MDD = abs(MDD)

        # print(f"The Maximum Drawdown is: {MDD}")
        
        return - abs_MDD + max_mdd
    
    def mdd_duration_constraint(
            self,
            weight: list,
        ):
        
        class RecoveryTime:
            def __init__(self, duration=0, start_index=0):
                self.duration = duration
                self.start_index = start_index

            def __repr__(self):
                return f"RecoveryTime(duration={self.duration}, start_index={self.start_index})"
        
        max_mdd_duration = self.strategy.max_mdd_duration   
        investments = self.portfolio.investments
        codes = list(investments.keys())
        weights = weight
        
        # Step 1: Compute the daily portfolio value
        portfolio_value = (self.universe.df_price[codes] * weights).sum(axis=1) 
        index_list = list(portfolio_value.index)
        
        peak = portfolio_value.iloc[0]
        peak_index = 0
        max_drawdown_recovery_time = RecoveryTime(0, 0)
        
        # Iterate through the portfolio values to calculate drawdown and recovery time
        for i in range(1, len(portfolio_value)):
            # Check if current value has recovered to the previous peak value
            if portfolio_value.iloc[i] >= peak and i > peak_index:
                # Calculate recovery time
                recovery_time = i - peak_index
                
                # Check if this is the longest recovery so far
                if recovery_time > max_drawdown_recovery_time.duration:
                    max_drawdown_recovery_time = RecoveryTime(recovery_time, peak_index)
                peak = portfolio_value.iloc[i]
                peak_index = i

        # print("start_date: ", index_list[max_drawdown_recovery_time.start_index])
        # print("end_date: ", index_list[max_drawdown_recovery_time.start_index+max_drawdown_recovery_time.duration])
        duration = max_drawdown_recovery_time.duration
        return -duration + max_mdd_duration
    
    
    def including_stocks_constraint(
        self,
        weight : list,
    ):
        stocks_to_be_included = self.strategy.stocks_to_be_included
        investments = self.portfolio.investments
        codes = list(investments.keys())

        eps = 1e-4 + 1e-5
        # print(stocks_to_be_included)
        if type(stocks_to_be_included) == list :
            ineq_constraints = []
            for stock in stocks_to_be_included:
                idx = codes.index(stock)
                ineq_constraints.append(weight[idx] - eps)
            ineq_constraints = tuple(ineq_constraints)
        else:
            idx = codes.index(stocks_to_be_included)
            ineq_constraints = weight[idx] - eps
        
        return ineq_constraints  # tuple of weight[stocks] > 0 

    
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
    
    
    def stocks_number_constraint(
        self,
        weight : list,
    ):
        eps = 1e-4
        max_stocks_number = self.strategy.max_stocks_number_constraint
        # Approximated extended cardinality constraint
        coefficient = 100000000
        approximated_num_of_stocks = [ 1 / ( 1 + np.e ** ( - (coefficient * ( w - eps )) ) ) for w in weight ]
        approximated_num_of_stocks = np.sum(approximated_num_of_stocks) 
        # print("num of stocks :", approximated_num_of_stocks)
        # print("min num of stocks :", max_stocks_number)
        return - approximated_num_of_stocks + max_stocks_number - eps  # (number of stocks) <= (max number of stocks)
    
    
    # def stock_ratio_constraint(
    #     self,
    #     weight : list
    # ):
    #     min_number_of_stocks = self.strategy.min_stock_ratio[0]
    #     min_ratio = self.strategy.min_stock_ratio[1]

    #     number_of_stocks = 0
    #     for w in weight:
    #         if w >= min_ratio:
    #             number_of_stocks += 1
    #     print("weight sum:", np.sum(weight))
    #     print("number of stocks: ", number_of_stocks)
    #     print("min_number_of_stocks: ", min_number_of_stocks)
    #     return number_of_stocks - min_number_of_stocks      
    
    def industry_ratio_constraint(
        self,
        weight : list,
    ):
        industry = self.strategy.min_industry_ratio[0]
        min_ratio = float(self.strategy.min_industry_ratio[1])
        stocks_list_of_industry = self.universe.get_stocks_of_industry(industry)
        
        investments = self.portfolio.get_weights_dict()
        codes = self.portfolio.get_stock_codes()
        
        weight_sum_of_industry_stocks = 0
        for stock in stocks_list_of_industry:
            idx = codes.index(stock)
            weight_sum_of_industry_stocks += weight[idx]
        
        return weight_sum_of_industry_stocks - min_ratio  # (weight of industry >= min ratio)
    
    
        
    def update_portfolio(
        self
    ) -> dict :
        
        
        self.expected_return = expected_returns.mean_historical_return(self.universe.df_price, frequency= self.universe.number_of_trading_days)
        self.covariance_matrix = risk_models.sample_cov(self.universe.df_price, frequency= self.universe.number_of_trading_days)
        
        # Select Constraints
        constraints = []
        if self.strategy.max_weights_sum != None :
            constraints.append({'type': 'eq', 'fun': self.weight_sum_constraint})
        if self.strategy.min_cumulative_return != None:
            constraints.append({'type': 'ineq', 'fun': self.return_constraint})
        if self.strategy.max_variance != None :
            constraints.append({'type': 'ineq', 'fun': self.variance_constraint})
        if self.strategy.max_mdd != None :
            constraints.append({'type': 'ineq', 'fun': self.mdd_constraint})
        if self.strategy.max_mdd_duration != None :
            constraints.append({'type': 'ineq', 'fun': self.mdd_duration_constraint})
        if self.strategy.stocks_to_be_included != None :
            constraints.append({'type': 'ineq', 'fun': self.including_stocks_constraint})
        if self.strategy.max_stocks_number_constraint != None :
            constraints.append({'type': 'ineq', 'fun': self.stocks_number_constraint})
        if self.strategy.min_stock_ratio != None :
            constraints.append({'type': 'ineq', 'fun': self.stock_ratio_constraint})
        if self.strategy.min_industry_ratio != None :
            constraints.append({'type': 'ineq', 'fun': self.industry_ratio_constraint})
        
        trial = 1
        while(1):
            # initial_weight = np.ones(len(self.expected_return)) / len(self.expected_return)
            initial_weight = np.random.rand(len(self.expected_return))
            initial_weight /= initial_weight.sum()  
            bounds = [(0, 1) for _ in range(len(self.expected_return))]

            result = minimize(self.objectivefn, initial_weight, method = 'SLSQP', constraints=constraints, bounds=bounds)

            weights = {}
            for i in range(len(self.stock_list)):
                weights[self.stock_list[i]] = result.x[i]
                

            self.portfolio.update_portfolio(weights)
            
            
            # To avoid local optima
            if trial > 50:
                print("No portfolio satisfies the constraints")
                break
            if self.strategy.max_weights_sum is not None and not weights_sum_constraint(self.portfolio, self.universe, self.strategy):
                trial += 1
                continue
            elif self.strategy.min_cumulative_return is not None and not cumulative_return_constraint(self.portfolio, self.universe, self.strategy):
                trial += 1
                continue
            elif self.strategy.max_variance is not None and not variance_constraint(self.portfolio, self.universe, self.strategy):
                trial += 1
                continue
            elif self.strategy.max_stocks_number_constraint is not None and not stocks_number_constraint(self.portfolio, self.universe, self.strategy):
                trial += 1
                continue
            elif self.strategy.min_stock_ratio is not None and not stock_ratio_constraint(self.portfolio, self.universe, self.strategy):
                trial += 1
                continue
            elif self.strategy.stocks_to_be_included is not None and not including_stocks_constraint(self.portfolio, self.universe, self.strategy):
                trial += 1
                continue
            elif self.strategy.min_industry_ratio is not None and not industry_ratio_constraint(self.portfolio, self.universe, self.strategy):
                trial += 1
                continue
            elif self.strategy.max_mdd is not None and not mdd_constraint(self.portfolio, self.universe, self.strategy):
                trial += 1
                continue
            elif self.strategy.max_mdd_duration is not None and not mdd_duration_constraint(self.portfolio, self.universe, self.strategy):
                trial += 1
                continue
            else:
                print("trial : ", trial)
                break
            
        print("Portfolio updated")
        print(f"Calculated objective: {self.compute_objective()}")

        return weights
    
