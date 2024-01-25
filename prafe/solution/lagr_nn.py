import os
import math 

import torch
from torch import nn
from torch.nn import MSELoss
from torch.nn.functional import one_hot, gumbel_softmax
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, random_split

import cvxpy as cp
from enum import IntEnum
import numpy as np


from tqdm import tqdm
from sklearn import metrics 
from prafe.solution.solution import Solution


class Lagrange_Neural_Net(nn.Module, Solution):

    def __init__(self, universe, portfolio, solution_name, method, N, K):
        '''
            universe: new_universe
            portfolio: new_portfolio
            solution_name: args.solution_name, 
            method: args.method
            N: len(index_stocks_list)
            K: K
        '''
        
        self.universe = universe
        self.portfolio = portfolio
        self.solution_name = solution_name
        self.method = method
        self.num_assets = N
        self.K = K
        self.new_return = np.array(self.universe.df_return)
        self.new_index = np.array(self.universe.df_index)
                
        super(Lagrange_Neural_Net, self).__init__()
        Solution.__init__(self, self.universe, self.portfolio)

        # build a neural network
        self.ann = nn.Sequential()
        self.ann.add_module("encoding_1", nn.Linear(self.num_assets, self.K))
        self.ann.add_module("hidden_1", nn.Linear(self.K, self.K))
        self.ann.add_module("decoding_1", nn.Linear(self.K, self.num_assets))
        
        # train option
        self.tau = 0.1
        self.epoch = 5
        self.train_ratio = 0.7
        self.batch_size = 24
        self.learning_rate = 1e-04
        self.criterion = MSELoss()
        
    
    
    def forward(self, new_return):
        
        x = new_return
        new_weight = gumbel_softmax(self.ann(x), tau=self.tau, hard=True)
        
        return new_weight
    
    
    # in practice, there's a training session in here via main code. 
    def update_portfolio(self) -> dict:
        
        new_x = self.new_return
        new_y = self.new_index
        
        # transform to torch tensor
        tensor_x = torch.Tensor(new_x)
        tensor_y = torch.Tensor(new_y)
        
        # split dataset into two parts (train, test)
        new_dataset = TensorDataset(tensor_x, tensor_y)
        data_size = len(new_dataset)
        train_size = int(data_size * self.train_ratio) 
        valid_size = int(data_size - train_size)
                
        train_dataset, valid_dataset = random_split(
            new_dataset, [train_size, valid_size]
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size
        )
        
        test_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size
        )
        
        # optimization
        opt = Adam(self.parameters(), self.learning_rate)
        
        weight = []
        min_loss = 100
        for epoch in range(0, self.epoch):
            loss_mean = []

            for i, data in enumerate(train_loader, 0):
                x, target = data
                self.train()

                pred = self(x).T

                opt.zero_grad()
                
                loss = self.criterion(x @ pred, target) 
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            print(f"[Train] Epoch: {epoch}, weight: {pred} Loss Mean: {np.mean(loss_mean)}")

            with torch.no_grad():
                loss_mean = []
                for i, data in enumerate(test_loader, 0):
                    x, target = data

                    self.eval()
                    
                    pred = self(x).T

                    loss = self.criterion(x @ pred, target) 
                    print(f"[Valid] number: {i}, weight: {pred}, loss: {loss}")

                    if min_loss > loss : 
                        # torch.save(
                        #     self.state_dict(),
                        #     os.path.join(
                        #         ckpt_path, "model.ckpt"
                        #     )
                        # )
                        weight = pred
                        print(f"weight: {pred}")
                        min_loss = loss
        print(f"========== Finished Epoch: {epoch} ============")
        
        return weight, min_loss