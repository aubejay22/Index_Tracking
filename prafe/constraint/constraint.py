import numpy as np
from prafe.portfolio import Portfolio
from prafe.universe import Universe
from prafe.objective import cumulative_return, variance, mdd, mdd_duration

eps = 1e-4 # 0.0001

def weights_sum(
        portfolio : Portfolio,
        universe : Universe,
    ):
    investments = portfolio.investments
    weights = list(investments.values())
    weights = np.array(weights)
    return np.sum(weights)


def weights_sum_constraint(
        portfolio : Portfolio,
        universe : Universe,
    ):
    max_weight_sum = 1
    if max_weight_sum is not None:
        investments = portfolio.investments
        weights = list(investments.values())
        weights = np.array(weights)
    else:
        return None
    return abs(np.sum(weights) - max_weight_sum) <= eps


def stocks_number_constraint(
        portfolio : Portfolio,
        universe : Universe,
        K : int,
    ) -> bool:

    max_number_of_stocks = K
    investments = portfolio.investments
    
    if max_number_of_stocks is not None :
        number_of_stocks = 0
        for stock in investments:
            if investments[stock] > eps:
                number_of_stocks += 1
    else:
        return None

    return number_of_stocks <= max_number_of_stocks 


# def stock_ratio_constraint(
#         portfolio : Portfolio,
#         universe : Universe,
#     ) -> bool:
    
#     if strategy.min_stock_ratio is not None :
#         min_number_of_stock = strategy.min_stock_ratio[0]
#         min_ratio = strategy.min_stock_ratio[1]
#         investments = portfolio.investments
#         number_of_stocks = 0
#         for stock in investments:
#             if investments[stock] >= min_ratio:
#                 number_of_stocks += 1
#     else:
#         return None

#     return number_of_stocks >= min_number_of_stock

