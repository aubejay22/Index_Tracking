import numpy as np
from prafe.universe import Universe
import json
import pickle

class Portfolio:
    def __init__(self, universe: Universe):
        self.universe = universe
        self.investments = {code: 0.0 for code in universe.stock_list}
        self.rewards = np.array([])

    def initialize_portfolio(self):
        for code in self.investments:
            self.investments[code] = 0.0

    def update_portfolio(self, weights: dict):
        for stock_code, weight in weights.items():
            self.investments[stock_code] = weight

    def get_weights(self) -> list:
        return list(self.investments.values())

    def get_weights_dict(self):
        return self.investments

    def get_stock_codes(self) -> list:
        return list(self.investments.keys())

    def get_rewards(self) -> np.ndarray:
        return self.rewards

    def save_portfolio(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.investments, f, indent=4)

    def save_portfolio_pickle(self, path: str):
        with open(path, 'wb') as handle:
            pickle.dump(self.investments, handle, protocol=pickle.HIGHEST_PROTOCOL)
