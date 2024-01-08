import numpy as np
from prafe.portfolio import Portfolio
from prafe.universe import Universe

class Strategy():
    
    def __init__(
        self,
        portfolio : Portfolio,
        universe : Universe
    ):
        self.portfolio = portfolio
        self.universe = universe
    
    
    
    
class GenerativeStrategy(Strategy) :

    def __init__(
            self, 
            args, 
            portfolio : Portfolio, 
            universe : Universe,
        ):
        super().__init__(portfolio, universe)

        # Objective
        self.objective = args.objective

        # Constraints
        self.max_weights_sum = args.max_weights_sum
        self.max_variance = args.max_variance
        self.max_mdd = args.max_mdd
        self.max_mdd_duration = args.max_mdd_duration
        self.min_cumulative_return = args.min_cumulative_return
        self.stocks_to_be_included = args.stocks_to_be_included
        self.max_stocks_number_constraint = args.max_stocks_number_constraint
        self.min_industry_ratio = args.min_industry_ratio
        self.min_stock_ratio = args.min_stock_ratio

        print("Current objective: ", args.objective)
        print("Current constraints: ")
        print("max_weights_sum: ", self.max_weights_sum) if self.max_weights_sum is not None else None
        print("max_variance: ", self.max_variance) if self.max_variance is not None else None
        print("max_mdd: ", self.max_mdd) if self.max_mdd is not None else None
        print("max_mdd_duration: ", self.max_mdd_duration) if self.max_mdd_duration is not None else None
        print("min_cumulative_return: ", self.min_cumulative_return) if self.min_cumulative_return is not None else None
        # print("max_stocks_number: ", self.max_stocks_number) if self.max_stocks_number is not None else None
        print("stocks_to_be_included: ", self.stocks_to_be_included) if self.stocks_to_be_included is not None else None
        print("max_stocks_number_constraint", self.max_stocks_number_constraint) if self.max_stocks_number_constraint is not None else None
        print("min_industry_ratio: ", self.min_industry_ratio) if self.min_industry_ratio is not None else None
        print("min_stock_ratio: ", self.min_stock_ratio) if self.min_stock_ratio is not None else None
        print("=====================================")


