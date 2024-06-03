import os
import math 

import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn import MSELoss, Parameter, Softmax
from torch.nn.functional import one_hot, gumbel_softmax, normalize
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, random_split

import cvxpy as cp
from enum import IntEnum
import numpy as np


from tqdm import tqdm
from sklearn import metrics 
from prafe.solution.solution import Solution


class SNN(nn.Module, Solution):

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
        self.stock_list = self.universe.stock_list
                
        super(SNN, self).__init__()
        Solution.__init__(self, self.universe, self.portfolio)
        
        # train option
        self.tau = 0.1
        self.epoch = 100
        self.train_ratio = 1
        self.batch_size = len(self.new_return)
        self.learning_rate = 1e-04
        self.criterion = MSELoss()
        self.softmax = Softmax()
        
        
        # parameter
        self.S = xavier_uniform_(torch.randn(self.K, self.num_assets))
        self.w_tilde = torch.zeros(self.num_assets)

        # build a neural network
        self.tilde_layer = nn.Sequential()
        self.tilde_layer.add_module("tilde_encoding_1", nn.Linear(self.num_assets, self.K))
        self.tilde_layer.add_module("tilde_hidden_1", nn.Linear(self.K, self.K))
        self.tilde_layer.add_module("tilde_decoding_1", nn.Linear(self.K, self.num_assets))
        
        self.S_layer = nn.Sequential()
        self.S_layer.add_module("S_encoding_1", nn.Linear(self.num_assets, self.K))
        self.S_layer.add_module("S_hidden_1", nn.Linear(self.K, self.K))
        self.S_layer.add_module("S_decoding_1", nn.Linear(self.K, self.num_assets))
        
        
    def QP(self):    
        """
            Quardratic Programming
            
        """ 
           
        # Define initial weight & Error
        initial_weight = cp.Variable(self.num_assets)
        error = self.new_return @ initial_weight - self.new_index
        
        # Define Objective & Constratins & Problem
        objective = cp.Minimize(cp.sum_squares(error))
        constraint = [cp.sum(initial_weight) == 1, initial_weight >= 0]
        problem = cp.Problem(objective, constraint)
        
        # Optimization
        problem.solve(solver='OSQP',verbose=True)
    
        return 0
    
    def forward(self):
                        
        self.w_tilde = self.tilde_layer(self.w_tilde)
        self.S = self.S_layer(self.S)
        
        # print(f"self.w_tilde_shape: {self.w_tilde.shape}")
        # print(f"self.S_shape: {self.S.shape}")
        
        w_hat = torch.exp(self.w_tilde)
        # print(f"self.w_hat_shape: {w_hat.shape}")
        
        pi = self.softmax(self.S / self.tau)
        z = gumbel_softmax(pi, tau=self.tau, hard=True)
        z = torch.sum(z, dim=-2)
        # print(z)
        # print(f"self.pi_shape: {pi.shape}")
        # print(f"self.z_shape: {z.shape}")
                
        w_upper = w_hat * z
        # print(f"self.w_upper: {w_upper.shape}")
        # w = normalize(w_upper, dim=0)
        
        w = w_upper / torch.sum(w_upper)
        
        # print(f"w.shape {w.shape}")
                
        # print(f"{torch.eq(w, 1).sum(dim=0)}")
        
        return w
    
    
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
        best_epoch = 0
        min_loss = 100
        for epoch in range(0, self.epoch):
            loss_mean = []

            for i, data in enumerate(train_loader, 0):
                x, target = data
                self.train()

                pred = self()
                
                opt.zero_grad()
                
                loss = self.criterion(x @ pred.T, target)
                        
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())
                
            # train_loss = np.mean(loss_mean)

            # if min_loss > train_loss:
            #     weight = pred.tolist()
            #     min_loss = train_loss
            #     best_epoch = epoch

                

            # # print(f"[Train] Epoch: {epoch}, weight: {pred} Loss Mean: {np.mean(loss_mean)}")

            with torch.no_grad():
                loss_mean = []
                for i, data in enumerate(train_loader, 0):
                    x, target = data

                    self.eval()
                    
                    pred = self()

                    loss = self.criterion(x @ pred.T, target) 
                    # print(f"[Valid] number: {i}, weight: {pred}, loss: {loss}")

                    if min_loss > loss : 
                        # torch.save(
                        #     self.state_dict(),
                        #     os.path.join(
                        #         ckpt_path, "model.ckpt"
                        #     )
                        # )
                        weight = pred.tolist()
                        # print(f"weight: {pred}")
                        min_loss = loss
        print(f"========== Finished Epoch: {best_epoch} loss: {min_loss} ============")
        
        stock2weight = {}
        for i in range(len(self.stock_list)):
            # stock2weight[self.stock_list[i]] = np.array(weight[i])
            stock2weight[self.stock_list[i]] = weight[i]
                
            # Update Portfolio & Calculate Error
            self.portfolio.update_portfolio(stock2weight)
        
        return stock2weight, min_loss