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

from prafe.solution import Solution

class MVO(Solution):
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
        
        
    def generate_constraint_included_stock(self, sector_mapper, sector_lower, sector_upper):
        for idx, stock in enumerate(self.strategy.stocks_to_be_included):
            sector_mapper[stock] = "unconditional"+str(idx)
            sector_lower["unconditional"+str(idx)] = 0.0001
            sector_upper["unconditional"+str(idx)] = 1.0
            
        return sector_mapper, sector_lower, sector_upper

    def update_portfolio(
        self
    ) -> dict :
        mu = expected_returns.mean_historical_return(self.universe.df_price, frequency= 252)
        # S = risk_models.CovarianceShrinkage(self.universe.df_price).ledoit_wolf()
        S = risk_models.sample_cov(self.universe.df_price, frequency= self.universe.number_of_trading_days)
        # ef = EfficientFrontier(mu, S, verbose = False)
        ef = EfficientFrontier(mu, S, solver = cp.CPLEX, verbose=False)
        if self.strategy.max_weights_sum != 1:
            raise NotImplementedError
        if self.strategy.max_variance is not None:
            ef.add_constraint(lambda _: objective_functions.portfolio_variance(_, S) <= self.strategy.max_variance)
        if self.strategy.min_cumulative_return is not None:
            ef.add_constraint(lambda _: objective_functions.portfolio_return(_, mu, negative=False) >= self.strategy.min_cumulative_return)
        if self.strategy.max_mdd is not None and mdd_constraint(self.portfolio, self.universe, self.strategy.max_mdd) == False:
            raise NotImplementedError
        if self.strategy.max_mdd_duration is not None and mdd_duration_constraint(self.portfolio, self.universe, self.strategy.max_mdd_duration) == False:
            raise NotImplementedError
        
        sector_mapper = {}
        sector_lower = {}
        sector_upper = {}
        
        # "특정 종목 포함" 제약조건
        if self.strategy.stocks_to_be_included != None:
            sector_mapper, sector_lower, sector_upper = self.generate_constraint_included_stock(sector_mapper, sector_lower, sector_upper)
        
        # "특정 산업군 제한" 제약조건
        if self.strategy.min_industry_ratio != None:
            industry = self.strategy.min_industry_ratio[0]
            min_industry_ratio = float(self.strategy.min_industry_ratio[1])
            stocks_list_of_industry = self.universe.get_stocks_of_industry(industry)
            for stock_code in stocks_list_of_industry:
                sector_mapper[stock_code] = industry
                
            sector_lower[industry] = min_industry_ratio
            sector_upper[industry] = 1.0
            
        ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
            
        # "종목 수 제한" 제약조건(n% 이상의 비중을 투자한 종목이 k개 이상)
        if self.strategy.min_stock_ratio != None:
            min_number_of_stock = self.strategy.min_stock_ratio[0]
            min_stock_ratio = self.strategy.min_stock_ratio[1]
            
            y = cp.Variable(len(ef.tickers), boolean=True)
            ef.add_constraint(lambda x: x <= y)
            ef.add_constraint(lambda x: y * min_stock_ratio <= x)
            ef.add_constraint(lambda x: cp.sum(y) >= min_number_of_stock)
            
        # "종목 수 제어" 제약조건(투자한 종목이 k개 이하)
        if self.strategy.max_stocks_number_constraint != None:
            booleans = cp.Variable(len(ef.tickers), boolean=True)
            ef.add_constraint(lambda x: x <= booleans)
            ef.add_constraint(lambda x: cp.sum(booleans) <= self.strategy.max_stocks_number_constraint)
        
        if self.objective == "cumulative_return":
            ef.convex_objective(objective_functions.portfolio_return, expected_returns = mu, negative=True)
        elif self.objective == "variance":
            ef.min_volatility()
        else:
            raise NotImplementedError
        weights = ef.clean_weights()
        self.portfolio.update_portfolio(weights)
        print("Portfolio updated")
        # ef.portfolio_performance(verbose=True)

        return weights