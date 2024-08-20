from prafe.portfolio import Portfolio
from prafe.universe import Universe
from prafe.objective import cumulative_return, variance, mdd, mdd_duration
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import objective_functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
class Evaluator():
    
    def __init__(
        self,
        universe : Universe,
        portfolio : Portfolio
    ):
        self.universe = universe
        self.portfolio = portfolio

    # TODO: implement evaluate functions that compute every metrics all at once
    def evaluate(self) -> dict:
        
        evaluation = {}
        metric_list = dir(self)
        for metic in metric_list:
            if metic == "evaluate" or metic == "__init__" or metic.startswith("_"):
                metric_list.remove(metic)
        for metric in metric_list:
            if callable(getattr(self, metric)):
                evaluation[metric] = getattr(self, metric)()
        
        return evaluation


    def calculate_cumulative_return(self):
        """
        Cumulative Return: 투자기간 동안의 포트폴리오의 누적 수익률
        """
        # print(self._calculate_cumulative_return())
        return self._calculate_cumulative_return()[-1]

    def _calculate_cumulative_return(self):
        """
        Cumulative Return: 투자기간 동안의 포트폴리오의 누적 수익률
        """
        
        # Fill NaN values with previous values(결측값은 전일 값 혹은 후일 값으로 대체)
        while self.universe.df_return.isnull().sum().sum() != 0:
            self.universe.df_return = self.universe.df_return.ffill().bfill()
        
        # Get the list of stocks and their weights in the portfolio
        investments = list(self.portfolio.investments.items())
        weight = list(self.portfolio.investments.values())
        # Creating an empty DataFrame to hold portfolio daily returns
        portfolio_daily_returns = pd.Series(0, index=self.universe.df_return.index)
        
        # # For each stock and its weight in the portfolio
        # for stock, weight in investments:
            
        #     # Add weighted daily return of each stock to portfolio daily return
        #     each_stock_daily_returns = self.universe.df_return[stock] * weight
        #     portfolio_daily_returns += each_stock_daily_returns
        
        each_stock_daily_returns = self.universe.df_return @ weight
        
        
        # Calculate the cumulative return for portfolio
        # cumulative_returns = (1 + portfolio_daily_returns).cumprod() - 1
        cumulative_returns = each_stock_daily_returns
        
        return cumulative_returns

    def calculate_CAGR(self):
        """
        Compound Annual Growth Rate: 연환산 수익률, 투자기간 동안의 포트폴리오의 연환산 평균 수익률
        일평균 수익률은 포트폴리오의 수익률이 과대평가되어 보일 수 있음 
        """
        investing_days = 365 # 임의값
        cumulative_return = self.calculate_cumulative_return()
        base = 1 + cumulative_return
        exponent = 252/investing_days
        
        annual_return = np.power(base, exponent) - 1
        
        return annual_return

    def calculate_AAR(self):
        """
        Annualized Average Return: 연환산 일별 평균 수익률, 투자기간 동안의 포트폴리오의 연환산 평균 수익률
        """
        
        # Fill NaN values with previous values(결측값은 전일 값 혹은 후일 값으로 대체)
        while self.universe.df_return.isnull().sum().sum() != 0:
            self.universe.df_return = self.universe.df_return.ffill().bfill()
        
        # Get the list of stocks and their weights in the portfolio
        investments = list(self.portfolio.investments.items())
        
        # Creating an empty DataFrame to hold portfolio daily returns
        portfolio_daily_returns = pd.Series(0, index=self.universe.df_return.index)
        
        # For each stock and its weight in the portfolio
        for stock, weight in investments:
            
            # Add weighted daily return of each stock to portfolio daily return
            each_stock_daily_returns = self.universe.df_return[stock] * weight
            portfolio_daily_returns += each_stock_daily_returns
            
        average = portfolio_daily_returns.sum()/len(portfolio_daily_returns)
        AAR = average*252 # 252 trading days in a year
        
        return AAR

    def calculate_AV(self):
        """
        Annualized Volatility(연환산 변동성): 투자 기간 동안의 포트폴리오 일별 수익률의 연환산 변동성
        
        note: portfolio volatility와 Annualized Volatility는 서로 volatility를 계산하는 공식이 다름.
        """
        
        # # Fill NaN values with previous values(결측값은 전일 값 혹은 후일 값으로 대체)
        # while self.universe.df_return.isnull().sum().sum() != 0:
        #     self.universe.df_return = self.universe.df_return.ffill().bfill()
        
        # # Get the list of stocks and their weights in the portfolio
        # investments = list(self.portfolio.investments.items())
        
        # # Creating an empty DataFrame to hold portfolio daily returns
        # portfolio_daily_returns = pd.Series(0, index=self.universe.df_return.index)
        
        # # For each stock and its weight in the portfolio
        # for stock, weight in investments:
            
        #     # Add weighted daily return of each stock to portfolio daily return
        #     each_stock_daily_returns = self.universe.df_return[stock] * weight
        #     portfolio_daily_returns += each_stock_daily_returns
            
        # variance = portfolio_daily_returns.var()
        # volatility = np.sqrt(variance)
        # AV = volatility*np.sqrt(252) 
        investments = self.portfolio.investments
        weights = list(investments.values())
        
        portfolio_returns = np.dot(self.universe.df_return, weights)
        daily_volatility = np.std(portfolio_returns)
        AV = daily_volatility * np.sqrt(252)
        
        return AV

    def calculate_LPM(self):
        """
        Lower Partial Moment: 투자기간 동안 포트폴리오의 하방 리스크를 측정하기 위한 지표
        
        threshold를 정해주어야 하는데, 우선은 0.001로 정한다. 
        (연 30%의 수익률을 기대한다면 매일 0.1%의 수익률을 얻으면 된다.)
        """
        
        # Specify the threshold
        threshold = 0.001
        
        # Fill NaN values with previous values(결측값은 전일 값 혹은 후일 값으로 대체)
        while self.universe.df_return.isnull().sum().sum() != 0:
            self.universe.df_return = self.universe.df_return.ffill().bfill()
        
        # Get the list of stocks and their weights in the portfolio
        investments = list(self.portfolio.investments.items())
        
        # Creating an empty DataFrame to hold portfolio daily returns
        portfolio_daily_returns = pd.Series(0, index=self.universe.df_return.index)
        
        # For each stock and its weight in the portfolio
        for stock, weight in investments:
            # Add weighted daily return of each stock to portfolio daily return
            each_stock_daily_returns = self.universe.df_return[stock] * weight
            portfolio_daily_returns += each_stock_daily_returns
            
        # Difference between threshold and returns for underperforming days
        diff = threshold - portfolio_daily_returns[threshold > portfolio_daily_returns]

        # Calculate average of the differences
        lpm = diff.mean()

        return lpm

    def calculate_VaR(self):
        """
        Value at Risk: 자산가치의 최대 예상 손실액, 
        
        임의로 z-value는 95% 신뢰구간에 상응하는 1.645로 사용한다.
        """
        
        z_value = 1.645
        while self.universe.df_return.isnull().sum().sum() != 0:
            self.universe.df_return = self.universe.df_return.ffill().bfill()

        # Get the list of stocks and their weights in the portfolio
        investments = list(self.portfolio.investments.items())

        # Creating an empty DataFrame to hold portfolio daily returns
        portfolio_daily_returns = pd.Series(0, index=self.universe.df_return.index)

        # For each stock and its weight in the portfolio
        for stock, weight in investments:
            
            # Add weighted daily return of each stock to portfolio daily return
            each_stock_daily_returns = self.universe.df_return[stock] * weight
            portfolio_daily_returns += each_stock_daily_returns
        
        average = portfolio_daily_returns.mean()
        Value_at_Risk = average - z_value * portfolio_daily_returns.std()

        return Value_at_Risk

    def calculate_Expected_Shortfall(self):
        """
        Expected Shortfall: 투자기간 중 VaR 수준을 넘어가는 포트폴리오의 expected return
        """
        # Fill NaN values with previous values(결측값은 전일 값 혹은 후일 값으로 대체)
        while self.universe.df_return.isnull().sum().sum() != 0:
            self.universe.df_return = self.universe.df_return.ffill().bfill()

        # Get the list of stocks and their weights in the portfolio
        investments = list(self.portfolio.investments.items())

        # Creating an empty DataFrame to hold portfolio daily returns
        portfolio_daily_returns = pd.Series(0, index=self.universe.df_return.index)

        # For each stock and its weight in the portfolio
        for stock, weight in investments:
            
            # Add weighted daily return of each stock to portfolio daily return
            each_stock_daily_returns = self.universe.df_return[stock] * weight
            portfolio_daily_returns += each_stock_daily_returns
            
        VaR = self.calculate_VaR()
        ES = portfolio_daily_returns[VaR > portfolio_daily_returns].mean()
        
        return ES

    def calculate_Information_Ratio(self):
        """
        정보비율(IR): 초과수익률(AER)/추적오차(ATE), 포트폴리오의 벤치마크 대비 성과에 대한 일관성을 측정하는 지표
        
        note: benchmark(e.g. kospi100)을 정해주어야 하는데, 우선은 0.001로 정한다. 
        (연 30%의 수익률을 기대한다면 매일 0.1%의 수익률을 얻으면 된다.)
        """
        
        # Specify the threshold
        benchmark = 0.001
        
        # Fill NaN values with previous values(결측값은 전일 값 혹은 후일 값으로 대체)
        while self.universe.df_return.isnull().sum().sum() != 0:
            self.universe.df_return = self.universe.df_return.ffill().bfill()
        
        # Get the list of stocks and their weights in the portfolio
        investments = list(self.portfolio.investments.items())
        
        # Creating an empty DataFrame to hold portfolio daily returns
        portfolio_daily_returns = pd.Series(0, index=self.universe.df_return.index)
        
        # For each stock and its weight in the portfolio
        for stock, weight in investments:
            
            # Add weighted daily return of each stock to portfolio daily return
            each_stock_daily_returns = self.universe.df_return[stock] * weight
            portfolio_daily_returns += each_stock_daily_returns
            
        # Difference in daily returns
        diff_returns = portfolio_daily_returns - benchmark
        
        # Annualized return differences
        avg_excess_return = diff_returns.mean() * 252

        # Annualized volatility
        tracking_error = diff_returns.std() * np.sqrt(252)

        # Information Ratio
        information_ratio = avg_excess_return / tracking_error

        return information_ratio

    ## for objective function
    def calculate_variance(self):
        investments = self.portfolio.investments
        codes = list(investments.keys())
        weights = list(investments.values())
        weights = np.array(weights)
        
        cov_matrix = np.cov(self.universe.df_return, rowvar=False)
        variance = np.dot(weights.T, np.dot(cov_matrix, weights))

        return variance
    
    
    def calculate_volatility(self):
        variance = self.calculate_variance()
        volatility = np.sqrt(variance)
        
        return volatility

    def calculate_sharpe_ratio(self):
        '''
        risk_free_ratio는 sharpe ratio 계산할 때, 따로 입력해주어야 하지만, input 형식을 맞추어야 하므로
        임의로 0.2를 지정해줌.
        '''
        investments = self.portfolio.investments
        codes = list(investments.keys())
        weights = list(investments.values())
        weights = np.array(weights)

        portfolio_return =  weights @ expected_returns.mean_historical_return(self.universe.df_price[codes], frequency= self.universe.number_of_trading_days)
        cov_matrix = np.cov(self.universe.df_return, rowvar=False)
            
        print(portfolio_return)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        print(portfolio_volatility)
        
        print(portfolio_return/portfolio_volatility)
        # print(weights@mu)
        # print(sigma)
        sign = 1
        risk_free_rate = 0
        sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility

        return sign * sharpe
        

    def calculate_mdd(self):
        
        investments = self.portfolio.investments
        codes = list(investments.keys())
        weights = list(investments.values())
        weights = np.array(weights)

        # Step 1: Compute the daily portfolio value
        portfolio_value = (self.universe.df_price[codes] * weights).sum(axis=1) 
 
        # Step 2: Compute the running maximum
        running_max = np.maximum.accumulate(portfolio_value)

        # Step 3: Compute the daily drawdown
        drawdown = portfolio_value / running_max - 1

        # Step 4: Compute the maximum drawdown
        MDD = drawdown.min()
        abs_MDD = abs(MDD)

        # print(f"The Maximum Drawdown is: {MDD}")
        return abs_MDD
    
    def calculate_recovery_time(self):
        class RecoveryTime:
            def __init__(self, duration=0, start_index=0):
                self.duration = duration
                self.start_index = start_index

            def __repr__(self):
                return f"RecoveryTime(duration={self.duration}, start_index={self.start_index})"
                
        investments = self.portfolio.investments
        codes = list(investments.keys())
        weights = np.array(list(investments.values()))
        
        # Step 1: Compute the daily portfolio value
        portfolio_value = (self.universe.df_price[codes] * weights).sum(axis=1) 
        index_list = list(portfolio_value.index)
        

        # Plotting the portfolio value
        # plt.figure(figsize=(20, 10))  # Set the figure size as desired
        # plt.plot(portfolio_value, label='Portfolio Value')
        # plt.title('Portfolio Value Over Time')
        # plt.xlabel('Time')
        # plt.ylabel('Value')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        # plt.savefig('portfolio_value.png')
        
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
        
