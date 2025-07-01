import pandas as pd
from datetime import datetime
import numpy as np

class Universe():
    
    
    def __init__(
        self,
        args,
        df_return : pd.DataFrame,
        df_index : pd.DataFrame,
    ) :
        self.args = args
        # # Log-Return
        # self.df_return = df_return.applymap(lambda x: x if x > -0.999999 else -0.999999)
        # self.df_return = np.log(1 + self.df_return)
        self.df_return = df_return
        self.df_index = df_index
        self.number_of_trading_days = len(df_return)
        self.stock_list = list(self.df_return.columns)

     

    def is_valid_date(self, date):
        if isinstance(date, datetime):
            date = pd.Timestamp(date)
        return date in self.df_return.index
            
    
    def _get_datetime_infos(self) -> list :
        datetime_infos = []
        
        for dataframe in [self.df_return, self.df_index]:
            if not dataframe.empty :
                dateframe_info = {'start' : dataframe.index[0],
                                  'end' : dataframe.index[-1]}
                datetime_infos.append(dateframe_info)
        
        return datetime_infos
    
    
    def _get_universe_datetime_info(
        self,
        join : str = 'inner'
    ) -> dict :
        datetime_infos = self._get_datetime_infos()
        
        # datetime_infos = self._get_datetime_infos()
        start_times = [datetime_info['start'] for datetime_info in datetime_infos]
        end_times = [datetime_info['end'] for datetime_info in datetime_infos]
        
        univere_datetime_info = {}
        
        if join == 'inner' :
            univere_datetime_info['start'] = min(start_times)
            univere_datetime_info['end'] = min(end_times)
        elif join == 'outer' :
            univere_datetime_info['start'] = max(start_times)
            univere_datetime_info['end'] = max(end_times)
        else :
            univere_datetime_info['start'] = None
            univere_datetime_info['end'] = None
            
        return univere_datetime_info
    
    
    def get_trimmed_universe_by_time(
        self,
        start_datetime : datetime,
        end_datetime : datetime
    ) :
        
        if type(start_datetime) != type(pd.Timestamp('now')):
            start_datetime = pd.Timestamp(start_datetime)
        if type(end_datetime) != type(pd.Timestamp('now')):
            end_datetime = pd.Timestamp(end_datetime)
        
        df_trimmed_returns = self.df_return.loc[start_datetime:end_datetime]
        df_trimmed_index = self.df_index.loc[start_datetime:end_datetime]

        common_index = df_trimmed_returns.index.intersection(df_trimmed_index.index)
        df_trimmed_returns = df_trimmed_returns.loc[common_index]
        df_trimmed_index = df_trimmed_index.loc[common_index]
        
        df_trimmed_index = (1 + df_trimmed_index).cumprod() - 1
        df_trimmed_returns = (1 + df_trimmed_returns).cumprod() - 1
        
        # print(df_trimmed_index)
        new_universe = Universe(args=self.args,
                                df_return=df_trimmed_returns,
                                df_index=df_trimmed_index)
                
        return new_universe
    
    
    def get_trimmed_universe_by_stocks(
        self,
        list_of_stock_codes : list = []
    )  :
        
        df_trimmed_returns = self.df_return[list_of_stock_codes]
        df_trimmed_index = self.df_index

        new_universe = Universe(args=self.args,
                                df_return=df_trimmed_returns,
                                df_index=df_trimmed_index)
        new_universe.stock_list = list_of_stock_codes

        return new_universe
    

    
    def get_mean_returns(
        self
    ) -> dict :

        series_mean = self.df_return.mean()
        return series_mean.to_dict()
    

