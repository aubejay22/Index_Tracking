import pandas as pd
from datetime import datetime
from collections import defaultdict
import torch
import numpy as np

class Universe():
    
    
    def __init__(
        self,
        args,
        df_price : pd.DataFrame,
        df_return : pd.DataFrame,
        df_index : pd.DataFrame,
    ) :
        self.args = args
        self.df_price = df_price
        # # Log-Return
        # self.df_return = df_return.applymap(lambda x: x if x > -0.999999 else -0.999999)
        # self.df_return = np.log(1 + self.df_return)
        self.df_return = df_return
        self.df_index = df_index
        self.number_of_trading_days = len(df_return)
        #self.universe_start_date = self._get_universe_start_date(join = 'inner')
        self.stock_list = list(self.df_return.columns)

        # Code name and type
        self.code_name = {}
        self.code_type = {}
        with open(args.data_path+'/code_name_type.tsv', 'r') as f:
            for line in f:
                code, name, type = line.split('\t')
                self.code_name[code] = name
                self.code_type[code] = type
                
        
        # Industry Type
        self.code_industry = {}
        industry_df = pd.read_excel(args.data_path+'/industry_information_by_NC.xlsx')
        trimmed_industry_df = industry_df[industry_df['Code'].isin(self.stock_list)]
        self.code_industry.update(trimmed_industry_df.set_index('Code')['WICS업종명(대)'].to_dict())


    def is_valid_date(self, date):
        if isinstance(date, datetime):
            date = pd.Timestamp(date)
        return date in self.df_price.index
            
    
    def _get_datetime_infos(self) -> list :
        datetime_infos = []
        
        for dataframe in [self.df_price, self.df_return, self.df_index]:
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
        
        df_trimmed_price = self.df_price.loc[start_datetime:end_datetime]
        df_trimmed_returns = self.df_return.loc[start_datetime:end_datetime]
        df_trimmed_index = self.df_index.loc[start_datetime:end_datetime]
        
        common_index = df_trimmed_price.index.intersection(df_trimmed_returns.index).intersection(df_trimmed_index.index)
        df_trimmed_price = df_trimmed_price.loc[common_index]
        df_trimmed_returns = df_trimmed_returns.loc[common_index]
        df_trimmed_index = df_trimmed_index.loc[common_index]
        
        df_trimmed_index = (1 + df_trimmed_index).cumprod() - 1
        df_trimmed_returns = (1 + df_trimmed_returns).cumprod() - 1
        
        # print(df_trimmed_index)
        new_universe = Universe(args = self.args,
                                df_price = df_trimmed_price,
                                df_return = df_trimmed_returns,
                                df_index = df_trimmed_index)
                
        return new_universe
    
    
    def get_trimmed_universe_by_stocks(
        self,
        list_of_stock_codes : list = []
    )  :
        
        df_trimmed_price = self.df_price[list_of_stock_codes]
        df_trimmed_returns = self.df_return[list_of_stock_codes]
        df_trimmed_index = self.df_index
        
        new_universe = Universe(args = self.args,
                                df_price = df_trimmed_price,
                                df_return = df_trimmed_returns,
                                df_index = df_trimmed_index)
        new_universe.stock_list = list_of_stock_codes

        return new_universe
    

    def get_name_type(
        self,
        stock_code : str = ''
    ):
        return self.code_name[stock_code], self.code_type[stock_code]
    
    
    def get_industry(
        self,
        stock_code : str = ''
    ):
        return self.code_industry[stock_code]
    
    
    def get_stocks_of_industry(
        self,
        industry_name : str,
    ):
        stocks_of_industry = [key for key, value in self.code_industry.items() if value == industry_name]
        return stocks_of_industry

    
    def get_mean_returns(
        self
    ) -> dict :

        series_mean = self.df_return.mean()
        return series_mean.to_dict()
    

    def get_start_price(
        self,
        stock_code : str = ''
    ) -> float:
        
        start_date = self.df_price.index[0]
        
        return self.df_price[stock_code][start_date]
    

    def get_price_dim(
        self,
        stock_list : list = []
    ) -> dict :
        
        price_dim = defaultdict(list)

        for stock_code in stock_list:
            price_dim[stock_code].append(self.df_price[stock_code].iloc[1])

        return price_dim

    
    def get_stock_infos(
        self,
        stock_code : str = ''
    ) -> dict :
            
        series_price = self.df_price[stock_code]
        series_return = self.df_return[stock_code]
        mean_return = series_return.mean()
        stock_multi_factor = stock_multi_factor.set_index('date')
        stock_multi_factor = stock_multi_factor.drop(columns = ['code'])
        stock_multi_factor = stock_multi_factor.loc[:, ~stock_multi_factor.columns.str.contains('^Unnamed')]
        
        stock_infos = {
            'name': self.code_name[stock_code],
            'type': self.code_type[stock_code],
            # 'industry': self.code_industry[stock_code],
            'price' : series_price,
            'return' : series_return,
            'mean_return' : mean_return,
        }
        
        return stock_infos
    
        
    def get_stocks_infos(
        self,
        list_of_stock_codes : list = []
    ) -> dict :
        
        stock_infos = {}
        
        for stock_code in list_of_stock_codes:
            series_price = self.df_price[stock_code]
            series_return = self.df_return[stock_code]
            mean_return = series_return.mean()
            stock_multi_factor = stock_multi_factor.set_index('date')
            stock_multi_factor = stock_multi_factor.drop(columns = ['code'])
            stock_multi_factor = stock_multi_factor.loc[:, ~stock_multi_factor.columns.str.contains('^Unnamed')]
        
            stock_infos[stock_code] = {
                'price' : series_price,
                'series_return' : series_return,
                'mean_return' : mean_return,
            }
        
        return stock_infos