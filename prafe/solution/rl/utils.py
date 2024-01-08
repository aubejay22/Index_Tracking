from typing import List
import numpy as np

from prafe.solution.rl.dataset import D4RLPortfolioDataset
from prafe.portfolio import Portfolio
from prafe.universe import Universe
from prafe.strategy import Strategy

from prafe.constraint.constraint import *


def get_dataset(
    universe : Universe, 
    constraints : List,
    weights : np.ndarray
) -> D4RLPortfolioDataset :
    
    # state : Universe
    # action : weight
    # constraint : constrait
    
    dataset = None
    
    return dataset
