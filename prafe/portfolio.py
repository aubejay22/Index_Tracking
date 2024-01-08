import numpy as np
from collections import defaultdict
from prafe.universe import Universe
import json
import pickle

class Portfolio():
    
    
    def __init__(
        self,
        universe : Universe
    ):
        self.investments = dict()   # { 'stock_code' : weight }
        self.rewards = np.array([])
        self.universe = universe

        for stock_code in universe.stock_list:
            self.investments[stock_code] = 0.0

    def initialize_portfolio(
        self,
    ):
        for stock_code in self.investments.keys():
            self.investments[stock_code] = 0.0
            
    def update_portfolio(
        self, 
        weights : dict
    ):
        for stock_code in weights.keys():
            self.investments[stock_code] = weights[stock_code]
        
    def get_weights(
        self
    ) -> list :
        
        return list(self.investments.values())

    def get_weights_dict(
        self
    ) -> list :
        
        return self.investments
    
    
    def get_stock_codes(
        self
    ) -> list:
        return list(self.investments.keys())
    
    
    def get_rewards(
        self
    ) -> np.ndarray :
        
        return self.rewards
    
    
    def save_portfolio(
        self,
        path : str
    ):
        with open(path, 'w') as f:
            json.dump(self.investments, f, indent=4)
            
    def save_portfolio_pickle(
        self,
        path: str
    ):
        with open(path, 'wb') as handle:
            pickle.dump(self.investments, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            
            
            