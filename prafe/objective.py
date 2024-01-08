from prafe.portfolio import Portfolio
from prafe.universe import Universe
from pypfopt import objective_functions
from pypfopt import risk_models
from pypfopt import expected_returns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def cumulative_return(
    portfolio : Portfolio,
    universe : Universe
):
    """
    Cumulative Return: 투자기간 동안의 포트폴리오의 누적 수익률
    """
    
    return _calculate_cumulative_return(portfolio, universe)[-1]

def _calculate_cumulative_return(
    portfolio : Portfolio,
    universe : Universe
):
    """
    Cumulative Return: 투자기간 동안의 포트폴리오의 누적 수익률
    """
    
    # Fill NaN values with previous values(결측값은 전일 값 혹은 후일 값으로 대체)
    while universe.df_return.isnull().sum().sum() != 0:
        universe.df_return = universe.df_return.ffill().bfill()
    
    # Get the list of stocks and their weights in the portfolio
    investments = list(portfolio.investments.items())
    
    # Creating an empty DataFrame to hold portfolio daily returns
    portfolio_daily_returns = pd.Series(0, index=universe.df_return.index)
    
    # For each stock and its weight in the portfolio
    for stock, weight in investments:
        
        # Add weighted daily return of each stock to portfolio daily return
        each_stock_daily_returns = universe.df_return[stock] * weight
        portfolio_daily_returns += each_stock_daily_returns
    
    # Calculate the cumulative return for portfolio
    cumulative_returns = (1 + portfolio_daily_returns).cumprod() - 1
    
    return cumulative_returns

def variance(
        portfolio : Portfolio,
        universe : Universe
    ):
    investments = portfolio.investments
    codes = list(investments.keys())
    weights = list(investments.values())
    weights = np.array(weights)

    S = risk_models.sample_cov(universe.df_price[codes], frequency= universe.number_of_trading_days)
    variance = objective_functions.portfolio_variance(weights, S)
    
    return variance

def mdd(
        portfolio : Portfolio,
        universe : Universe
    ):
    
    investments = portfolio.investments
    codes = list(investments.keys())
    weights = list(investments.values())
    weights = np.array(weights)
    
    # Step 1: Compute the daily portfolio value
    portfolio_value = (universe.df_price[codes] * weights).sum(axis=1) 
    
    # Step 2: Compute the running maximum
    running_max = np.maximum.accumulate(portfolio_value)

    # Step 3: Compute the daily drawdown
    drawdown = portfolio_value / running_max - 1.0

    # Step 4: Compute the maximum drawdown
    MDD = drawdown.min()
    abs_MDD = abs(MDD)

    # print(f"The Maximum Drawdown is: {MDD}")
    return abs_MDD

def mdd_duration(
        portfolio : Portfolio,
        universe : Universe
    ):
    
    class RecoveryTime:
        def __init__(self, duration=0, start_index=0):
            self.duration = duration
            self.start_index = start_index

        def __repr__(self):
            return f"RecoveryTime(duration={self.duration}, start_index={self.start_index})"
                
    investments = portfolio.investments
    codes = list(investments.keys())
    weights = np.array(list(investments.values()))
    
    # Step 1: Compute the daily portfolio value
    portfolio_value = (universe.df_price[codes] * weights).sum(axis=1) 
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
    return max_drawdown_recovery_time.duration

def sharpe_ratio(
        portfolio : Portfolio,
        universe : Universe
    ):
    '''
    risk_free_ratio는 sharpe ratio 계산할 때, 따로 입력해주어야 하지만, input 형식을 맞추어야 하므로
    임의로 0.2를 지정해줌.
    '''
    investments = portfolio.investments
    codes = list(investments.keys())
    weights = list(investments.values())
    weights = np.array(weights)

    S = risk_models.sample_cov(universe.df_price[codes], frequency= universe.number_of_trading_days)
    mu = expected_returns.mean_historical_return(universe.df_price[codes], frequency= universe.number_of_trading_days)
    variance = objective_functions.portfolio_variance(weights, S)
    sigma = np.sqrt(variance)
    
    sign = 1
    risk_free_rate = 0.2
    sharpe = (weights @ mu - risk_free_rate) / sigma
    
    return sign * sharpe