import numpy as np
from prafe.portfolio import Portfolio
from prafe.universe import Universe
from prafe.strategy import Strategy, GenerativeStrategy
from prafe.objective import cumulative_return, variance, mdd, mdd_duration
from prafe.constraint.constraint import weights_sum_constraint, variance_constraint, mdd_constraint, mdd_duration_constraint, cumulative_return_constraint, stocks_number_constraint, industry_ratio_constraint, stock_ratio_constraint

from sklearn.model_selection import ParameterGrid, ParameterSampler
import time
import random
import torch
import matplotlib.pyplot as plt
from pykeops.torch import LazyTensor
from kneed import KneeLocator

use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
device_id = "cuda:1" if use_cuda else "cpu"

eps = 1e-6

class Solution():

    def __init__(
        self,
        universe : Universe,
        portfolio : Portfolio,
        strategy: GenerativeStrategy
        ):
        self.stock_list = universe.stock_list
        self.portfolio = portfolio
        self.universe = universe
        self.strategy = strategy

        self.objective = strategy.objective
        self.initial_weights = portfolio.get_weights()

    def compute_objective(
        self
    ) -> float :
        
        if self.objective == "cumulative_return":
            return cumulative_return(self.portfolio, self.universe)
        elif self.objective == "variance":
            return variance(self.portfolio, self.universe)
        elif self.objective == "mdd":
            return mdd(self.portfolio, self.universe)
        elif self.objective == "mdd_duration":
            return mdd_duration(self.portfolio, self.universe)
        else:
            raise NotImplementedError

    def update_portfolio(
        self
    ) -> dict :
        
        # TODO: Implement the update portfolio function
        weights = {}
        for stock_code in self.stock_list:
            weights[stock_code] = 0.0
        
        self.portfolio.update_portfolio(weights)
        print("Portfolio updated")
        print(f"Calculated objective: {self.compute_objective()}")

        return weights
    
    def update_rewards(
        self
    ) -> np.ndarray :
        
        rewards = np.array([])
        self.portfolio.rewards = rewards
        
        return self.portfolio.get_rewards()
    
    def _does_satisfy_constraints(self) -> bool:
        if self.strategy.max_weights_sum is not None and weights_sum_constraint(self.portfolio, self.universe, self.strategy.max_weights_sum) == False:
            return False
        if self.strategy.max_variance is not None and variance_constraint(self.portfolio, self.universe, self.strategy.max_variance) == False:
            return False
        if self.strategy.max_mdd is not None and mdd_constraint(self.portfolio, self.universe, self.strategy.max_mdd) == False:
            return False
        if self.strategy.max_mdd_duration is not None and mdd_duration_constraint(self.portfolio, self.universe, self.strategy.max_mdd_duration) == False:
            return False
        if self.strategy.min_cumulative_return is not None and cumulative_return_constraint(self.portfolio, self.universe, self.strategy.min_cumulative_return) == False:
            return False
        if self.strategy.max_stocks_number is not None and stocks_number_constraint(self.portfolio, self.universe, self.strategy.max_stocks_number) == False:
            return False
        # if self.strategy.max_industry_ratio is not None and industry_ratio_constraint(self.portfolio, self.universe, self.strategy.max_industry_ratio) == False:
        #     return False
        # if self.strategy.max_stock_ratio is not None and stock_ratio_constraint(self.portfolio, self.universe, self.strategy.max_stock_ratio) == False:
        #     return False
        return True


class DecisionTransformer(Solution):
    """
    
    """
    def __init__(
        self,
        universe : Universe,
        portfolio : Portfolio,
        strategy : Strategy
    ) :
        super().__init__(universe, portfolio, strategy)


    def update_portfolio(
        self
    ) -> dict :
         
        weights = {}
        
        return weights

# class HeuristicSearch(Solution):
#     """
#     HeuristicSearch is a class that implements the heuristic matching function with grid-search algorithm.
#     """

#     def __init__(
#         self,
#         args,
#         universe : Universe,
#         portfolio : Portfolio,
#         strategy: GenerativeStrategy
#         ):
#         super().__init__(universe, portfolio, strategy)
#         self.stock_list = universe.stock_list
#         self.portfolio = portfolio
#         self.universe = universe
#         self.strategy = strategy

#         self.objective = strategy.objective
#         self.initial_weights = portfolio.investments
#         self.search_method = args.search_method
#         self.num_clusters = args.num_clusters
#         self.args = args


#     def compute_objective(self):
#         return super().compute_objective()
    

#     def _does_satisfy_constraints(self) -> bool:
#         return super()._does_satisfy_constraints()

    

#     def grid_search(self, candidates, min_price, budget):
#         print("Grid search...")
#         start_time = time.time()
#         candidate_weights = []
#         candidate_objectives = []

#         price = {}
#         bins = {}

#         for stock_code in candidates:
#             price[stock_code] = self.universe.get_start_price(stock_code)
#             step = min_price / price[stock_code] if min_price / price[stock_code] > 1 else 1
#             bins[stock_code] = np.arange(0, budget / price[stock_code], step, dtype=float)

#         # Grid search
#         grids = ParameterGrid(bins)
        
#         # for grid in grids:
#         for grid in grids:
#             weights = {}
#             for stock_code in grid.keys():
#                 weights[stock_code] = grid[stock_code]
#             factor = 1.0 / (sum(weights.values()) + eps)
#             weights_norm = {k: v * factor for k, v in weights.items()}

#             current_budget = 0
#             for stock_code in weights.keys():
#                 current_budget += weights[stock_code] * price[stock_code]

#             self.portfolio.initialize_portfolio()
#             self.portfolio.update_portfolio(weights_norm)
#             if self._does_satisfy_constraints() and current_budget <= budget:
#                 candidate_weights.append(weights_norm)
#                 candidate_objectives.append(self.compute_objective())
        
#         # Select the best weights
#         if self.objective == "cumulative_return":
#             best_weights = candidate_weights[np.argmax(candidate_objectives)]
#         else:
#             best_weights = candidate_weights[np.argmin(candidate_objectives)]

#         end_time = time.time()
#         print(f"Time elapsed: {end_time - start_time} seconds ({(end_time - start_time) / 60} minutes)")
#         return best_weights, candidate_weights, candidate_objectives
    

#     def random_search(self, candidates, min_price, budget, n_iter=10):
#         print("Random search...")
#         start_time = time.time()
#         candidate_weights = []
#         candidate_objectives = []
#         candidate_budgets = []

#         # for comb in all_combinations:
#         price = {}
#         bins = {}

#         # Adjusting the search space
#         for stock_code in candidates:
#             price[stock_code] = self.universe.get_start_price(stock_code)
#             step = min_price / price[stock_code] if min_price / price[stock_code] > 1 else 1
#             bins[stock_code] = np.arange(0, budget / price[stock_code], step, dtype=int)
        
#         # Random search
#         rng = np.random.RandomState(0)
#         samples = list(ParameterSampler(bins, n_iter=n_iter, random_state=rng))
    
#         for sample in samples:
#             total_budget = 0
#             for stock_code in sample.keys():
#                 total_budget += sample[stock_code] * price[stock_code]

#             # Normalize the weights
#             x = np.array(list(sample.values()))
#             x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
#             weights = {}
#             for i, stock_code in enumerate(sample.keys()):
#                 weights[stock_code] = x_norm[i]
                
#             self.portfolio.initialize_portfolio()
#             self.portfolio.update_portfolio(weights)
#             if self._does_satisfy_constraints():
#                 candidate_weights.append(weights)
#                 candidate_objectives.append(self.compute_objective())
#                 candidate_budgets.append(total_budget)
        
#         # Select the best weights
#         if len(candidate_weights) == 0:
#             print("No feasible solution found")
#             return self.initial_weights
        
#         if self.objective == "cumulative_return":
#             best_weights = candidate_weights[np.argmax(candidate_objectives)]
#         else:
#             best_weights = candidate_weights[np.argmin(candidate_objectives)]

#         end_time = time.time()
#         print(f"Time elapsed: {end_time - start_time} seconds ({(end_time - start_time) / 60} minutes)")
#         return best_weights
    

#     def clustering(self, x, K=10, Niter=10, verbose=True):
#         """
#         implements Lloyd's algorithm for Cosine similarity metric.
#         reference: https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html
#         """
#         start = time.time()
#         N, D = x.shape  # Number of samples, dimension of the ambient space

#         c = x[:K, :].clone()  # Simplistic initialization for the centroids
#         # Normalize the centroids for the cosine similarity:
#         c = torch.nn.functional.normalize(c, dim=1, p=2)

#         x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
#         c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

#         # K-means loop:
#         # - x  is the (N, D) point cloud,
#         # - cl is the (N,) vector of class labels
#         # - c  is the (K, D) cloud of cluster centroids
#         for i in range(Niter):
#             # E step: assign points to the closest cluster -------------------------
#             S_ij = x_i | c_j  # (N, K) symbolic Gram matrix of dot products
#             cl = S_ij.argmax(dim=1).long().view(-1)  # Points -> Nearest cluster

#             # M step: update the centroids to the normalized cluster average: ------
#             # Compute the sum of points per cluster:
#             c.zero_()
#             c.scatter_add_(0, cl[:, None].repeat(1, D), x)

#             # Normalize the centroids, in place:
#             c[:] = torch.nn.functional.normalize(c, dim=1, p=2)

#         # Select stocks in each cluster with similarity
#         cl = cl.cpu().numpy()
#         max_index = []
#         for i in range(0, cl.max() + 1):
#             index = np.where(cl == i)[0]
#             current_max = 0
#             for j in index.tolist():
#                 sim = torch.cosine_similarity(x[j], c[i], dim=0)
#                 if sim > current_max:
#                     current_max = sim
#                     current_index = j
#             max_index.append(current_index)

#         if verbose:  # Fancy display -----------------------------------------------
#             if use_cuda:
#                 torch.cuda.synchronize()
#             end = time.time()
#             print(
#                 f"K-means for the cosine similarity with {N:,} points in dimension {D:,}, K = {K:,}:"
#             )
#             print(
#                 "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
#                     Niter, end - start, Niter, (end - start) / Niter
#                 )
#             )

#         return cl, c, max_index

    
#     def inertia_(self, x, cl, c):
#         """Computes the inertia of the K-means clustering given a dataset x,
#         a clustering cl and the cluster centroids c.
#         """
#         x_i = torch.Tensor(x.view(len(x), 1, -1))  # (N, 1, D) samples
#         c_j = torch.Tensor(c.view(1, len(c), -1))  # (1, K, D) centroids
#         cl = torch.LongTensor(cl)  # (N,) labels
#         sum_of_squared_dist = ((x_i - c_j) ** 2).sum(dim=-1).sum(dim=-1)
#         return sum_of_squared_dist.scatter_add_(0, cl, torch.ones(len(x)))


#     def update_portfolio(self, min_price=None, budget=None, num_stocks=None):

#         min_price = self.args.min_price if min_price is None else min_price
#         budget = self.args.budget if budget is None else budget
#         num_stocks = self.args.max_stocks_number if num_stocks is None else num_stocks

#         print("Selecting stocks...")
#         start_time = time.time()

#         stock_embeddings = self.universe.get_stock_embeddings(self.stock_list)
#         stock_embeddings = torch.stack(list(stock_embeddings.values())) # (num_stocks, date_length, embedding_dim)
#         # stock_embeddings = stock_embeddings.mean(dim=1) # (num_stocks, embedding_dim)
#         price_emb = stock_embeddings[:, :, 0] # (num_stocks, date_length)
#         norm_price_emb = torch.nn.functional.normalize(price_emb, dim=1, p=2)
#         multi_emb = stock_embeddings.mean(dim=1)[:, 1:] # (num_stocks, embedding_dim)
#         stock_embeddings = torch.cat([norm_price_emb, multi_emb], dim=1) # (num_stocks, date_length + embedding_dim)

#         # normalize
#         stock_embeddings_norm = torch.nn.functional.normalize(stock_embeddings, dim=1, p=2)

#         # if self.num_clusters is not None:
#         #     cl, centroids, max_index = self.clustering(stock_embeddings_norm, K=self.num_clusters)

#         sse = []
#         for k in range(2, self.num_clusters + 1):
#             cl, centroids, _ = self.clustering(stock_embeddings_norm, K=k)
#             inertia = self.inertia_(stock_embeddings_norm, cl, centroids)
#             sse.append(inertia.sum().item())
#             print(f"K = {k}, inertia = {inertia.sum().item()}")

#         # Plot the elbow
#         plt.style.use("fivethirtyeight")
#         plt.plot(range(2, self.num_clusters + 1), sse)
#         plt.xticks(range(2, self.num_clusters + 1))
#         plt.xlabel("Number of Clusters")
#         plt.ylabel("SSE")
#         plt.savefig("elbow.png")
#         plt.close()

#         kl = KneeLocator(range(2, self.num_clusters + 1), sse, curve="convex", direction="increasing")
#         print(f"Knee point: {kl.knee}")

#         # all_combinations = self.random_combinations(self.stock_list, num_stocks, iters=100)
#         cl, centroids, max_index = self.clustering(stock_embeddings_norm, K=kl.knee)
#         candidates = [self.stock_list[i] for i in max_index]

#         clusteing_result = {'class': cl, 'centroids': centroids, 'max_index': max_index, 'candidates': candidates}
#         torch.save(clusteing_result, "clustering_result.pt")

        
#         # if self.search_method == "grid_search":
#         best_weights, candidate_weights, candidate_obj = self.grid_search(candidates, min_price, budget)
#         # elif self.search_method == "random_search":
#         #     best_weights = self.random_search(candidates, min_price, budget)
#         # else:
#         #     raise NotImplementedError

#         self.portfolio.update_portfolio(best_weights)
#         print("Portfolio updated")
#         print(f"Calculated objective: {self.compute_objective()}")

#         end_time = time.time()
#         print("Heuristic search end ! =============================")
#         print(f"Time elapsed: {end_time - start_time} seconds ({(end_time - start_time) / 60} minutes)")

#         return best_weights
