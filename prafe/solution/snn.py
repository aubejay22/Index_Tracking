import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn import MSELoss, Softmax
from torch.nn.functional import gumbel_softmax
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, random_split

import numpy as np

class SNN(nn.Module):
    def __init__(self, universe, portfolio, solution_name, method, N, K):
        super().__init__()
        self.universe = universe
        self.portfolio = portfolio
        self.solution_name = solution_name
        self.method = method
        self.num_assets = N
        self.K = K
        self.new_return = np.array(self.universe.df_return)
        self.new_index = np.array(self.universe.df_index)
        self.stock_list = self.universe.stock_list

        self.tau = 0.1
        self.epoch = 100
        self.train_ratio = 1
        self.batch_size = len(self.new_return)
        self.learning_rate = 1e-4
        self.criterion = MSELoss(reduction='sum')
        self.softmax = Softmax()

        self.S = xavier_uniform_(torch.randn(self.K, self.num_assets))
        self.w_tilde = torch.zeros(self.num_assets)

        self.tilde_layer = nn.Sequential(
            nn.Linear(self.num_assets, self.K),
            nn.Linear(self.K, self.K),
            nn.Linear(self.K, self.num_assets),
        )
        self.S_layer = nn.Sequential(
            nn.Linear(self.num_assets, self.K),
            nn.Linear(self.K, self.K),
            nn.Linear(self.K, self.num_assets),
        )

    def objective_function(self, weight):
        error = self.new_return @ weight - self.new_index
        return np.sum(error ** 2)

    def forward(self):
        self.w_tilde = self.tilde_layer(self.w_tilde)
        self.S = self.S_layer(self.S)
        w_hat = torch.exp(self.w_tilde)
        pi = self.softmax(self.S / self.tau)
        z = gumbel_softmax(pi, tau=self.tau, hard=True)
        z = torch.sum(z, dim=-2)
        w_upper = w_hat * z
        w = w_upper / torch.sum(w_upper)
        return w

    def update_portfolio(self):
        tensor_x = torch.Tensor(self.new_return)
        tensor_y = torch.Tensor(self.new_index)
        dataset = TensorDataset(tensor_x, tensor_y)
        data_size = len(dataset)
        train_size = int(data_size * self.train_ratio)
        valid_size = data_size - train_size
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size)

        opt = Adam(self.parameters(), self.learning_rate)

        weight = []
        min_loss = 100
        for _ in range(self.epoch):
            for x, target in train_loader:
                self.train()
                pred = self()
                opt.zero_grad()
                loss = self.criterion(x @ pred.T, target)
                loss.backward()
                opt.step()

            with torch.no_grad():
                for x, target in valid_loader:
                    self.eval()
                    pred = self()
                    loss = self.criterion(x @ pred.T, target)
                    if min_loss > loss:
                        weight = pred.tolist()
                        min_loss = loss

        stock2weight = {self.stock_list[i]: weight[i] for i in range(len(self.stock_list))}
        self.portfolio.update_portfolio(stock2weight)
        self.optimal_error = self.objective_function(weight)
        return stock2weight, min_loss
