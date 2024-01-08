import numpy as np
from prafe.portfolio import Portfolio
from prafe.universe import Universe
from prafe.objective import cumulative_return, variance, mdd, mdd_duration
from prafe.strategy import GenerativeStrategy

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
        strategy : GenerativeStrategy,
    ):
    max_weight_sum = strategy.max_weights_sum
    if max_weight_sum is not None:
        investments = portfolio.investments
        weights = list(investments.values())
        weights = np.array(weights)
    else:
        return None
    return abs(np.sum(weights) - max_weight_sum) <= eps


def variance_constraint(
        portfolio : Portfolio,
        universe : Universe,
        strategy : GenerativeStrategy,
    ) -> bool:
    max_variance = strategy.max_variance
    if max_variance is not None :
        var_out = variance(portfolio, universe)
    else :
        return None
    return (var_out - max_variance) <= eps


def mdd_constraint(
        portfolio : Portfolio,
        universe : Universe,
        strategy : GenerativeStrategy,
    ) -> bool:
    max_mdd = strategy.max_mdd
    if max_mdd is not None :
        mdd_out = mdd(portfolio, universe)
    else:
        return None
    return (mdd_out - max_mdd) <= eps


def mdd_duration_constraint(
        portfolio : Portfolio,
        universe : Universe,
        strategy : GenerativeStrategy,
    ) -> bool:
    max_mdd_duration = strategy.max_mdd_duration
    if max_mdd_duration is not None:
        duration_out = mdd_duration(portfolio, universe)
    else:
        return None
    return (duration_out - max_mdd_duration) <= eps
    

def cumulative_return_constraint(
        portfolio : Portfolio,
        universe : Universe,
        strategy : GenerativeStrategy,
    ) -> bool:
    min_cumulative_return = strategy.min_cumulative_return
    if min_cumulative_return is not None:
        cumulative_return_out = cumulative_return(portfolio, universe)
    else : 
        return None
    return (cumulative_return_out - min_cumulative_return) >= -eps


def stocks_number_constraint(
        portfolio : Portfolio,
        universe : Universe,
        strategy : GenerativeStrategy,
    ) -> bool:

    max_number_of_stocks = strategy.max_stocks_number_constraint
    investments = portfolio.investments
    
    if max_number_of_stocks is not None :
        number_of_stocks = 0
        for stock in investments:
            if investments[stock] > eps:
                number_of_stocks += 1
    else:
        return None

    return number_of_stocks <= max_number_of_stocks 


def stock_ratio_constraint(
        portfolio : Portfolio,
        universe : Universe,
        strategy : GenerativeStrategy,
    ) -> bool:
    
    if strategy.min_stock_ratio is not None :
        min_number_of_stock = strategy.min_stock_ratio[0]
        min_ratio = strategy.min_stock_ratio[1]
        investments = portfolio.investments
        number_of_stocks = 0
        for stock in investments:
            if investments[stock] >= min_ratio:
                number_of_stocks += 1
    else:
        return None

    return number_of_stocks >= min_number_of_stock


def industry_ratio_constraint(
        portfolio : Portfolio,
        universe : Universe,
        strategy : GenerativeStrategy,
    ) -> bool:
    investments = portfolio.get_weights_dict()
    codes = portfolio.get_stock_codes()
    weight = portfolio.get_weights()
    
    if strategy.min_industry_ratio is not None:
        industry = strategy.min_industry_ratio[0]
        min_ratio = float(strategy.min_industry_ratio[1])
        stocks_list_of_industry = universe.get_stocks_of_industry(industry)

        weight_sum_of_industry_stocks = 0
        for stock in stocks_list_of_industry:
            idx = codes.index(stock)
            weight_sum_of_industry_stocks += weight[idx]

        return weight_sum_of_industry_stocks >= min_ratio - eps
    else:
        return None


def including_stocks_constraint(
        portfolio : Portfolio,
        universe : Universe,
        strategy : GenerativeStrategy
    ) :
    stocks = strategy.stocks_to_be_included
    investments = portfolio.investments
    
    include = True
    if type(stocks) == list :
        for stock in stocks:
            if investments[stock] < eps:
                # print(stock)
                # print(investments[stock])
                include = False
    elif type(stocks) == str :
        if investments[stocks] < eps :
            # print(stocks)
            # print(investments[stocks])
            include = False  
    # None
    else:
        return None
                
    return include
    
    