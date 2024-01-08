from prafe.strategy import GenerativeStrategy
from prafe.solution import Solution
from prafe.portfolio import Portfolio
from prafe.universe import Universe
import time
import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler
import torch
from pykeops.torch import LazyTensor
import matplotlib.pyplot as plt
from kneed import KneeLocator
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
device_id = "cuda:1" if use_cuda else "cpu"

eps = 1e-6

class HeuristicSearch(Solution):
    
    def __init__(
        self,
        args,
        universe : Universe,
        portfolio : Portfolio,
        strategy: GenerativeStrategy
        ):
        super().__init__(universe, portfolio, strategy)
        self.stock_list = universe.stock_list
        self.portfolio = portfolio
        self.universe = universe
        self.strategy = strategy

        self.objective = strategy.objective
        self.initial_weights = portfolio.investments
        self.search_method = args.search_method
        self.num_clusters = args.num_clusters
        self.args = args
        
        # Cluster stocks
        print("Clustering stocks...")
        stock_embeddings_norm = self._compute_stock_embeddings()
        print(self.args.do_clustering)
        if self.args.do_clustering:
            kl, sse = self._find_elbow()
            cl, centroids, max_index = self._clustering(stock_embeddings_norm, K=kl.knee)
        else:
            cl, centroids, max_index = self._clustering(stock_embeddings_norm, K=self.num_clusters)
        self.candidates = [self.stock_list[i] for i in max_index]
        
        # clusteing_result = {
        #                         'stock_embeddings': stock_embeddings_norm, 
        #                         'class': cl, 
        #                         'centroids': centroids, 
        #                         'max_index': max_index, 
        #                         'candidates': candidates,
        #                         'sse': sse
        #                     }
        # torch.save(clusteing_result, "clustering_result.pt")
        
        # handle the case where stock in target industry is not in the candidates
        if self.args.min_industry_ratio is not None:
            target_industry = self.args.min_industry_ratio[0]
            for stock_code, industry in self.universe.code_industry.copy().items():
                if industry == target_industry:
                    idx = self.stock_list.index(stock_code)
                    cluster_num = cl[idx]
                    center_stock_num = max_index[cluster_num]
                    center_stock_code = self.stock_list[center_stock_num]
                    self.universe.code_industry[center_stock_code] = target_industry

    def compute_objective(self):
        return super().compute_objective()
    

    def _does_satisfy_constraints(self) -> bool:
        return super()._does_satisfy_constraints()
    
    #  This problem is known as "Multinomial Coefficient".
    def _generate_investments(self, stock_list, step, total_budget):
        num_stocks = len(stock_list)
        
        # Define a recursive function to handle the partitioning.
        def _partition(number, count, limit):
            if count == 1:
                if 0 <= number <= limit:
                    yield (number,)
            else:
                for i in range(0, min(number, limit) + 1, step):
                    for result in _partition(number - i, count - 1, limit):
                        yield (i,) + result

        # Generate partitions.
        partitions = list(_partition(total_budget, num_stocks, total_budget))

        # Convert partitions into a list of dictionaries representing investments.
        investments = []
        for part in partitions:
            investment_dict = {stock_code: part[idx] for idx, stock_code in enumerate(stock_list)}
            investments.append(investment_dict)

        return investments

    def _grid_search(self, candidates, min_price, budget):
        print("Grid search...")
        start_time = time.time()
        candidate_weights = []
        candidate_objectives = []
        non_satisfied_weights = []
        non_satisfied_objectives = []
        price = {}
        bins = {}

        for stock_code in candidates:
            price[stock_code] = self.universe.get_start_price(stock_code)
            step = min_price / price[stock_code] if min_price / price[stock_code] > 1 else 1
            bins[stock_code] = np.arange(0, budget / price[stock_code], step, dtype=float)
        
        norm_budget = int(budget // min_price)
        norm_step = 1
        
        grids = self._generate_investments(list(bins.keys()), norm_step, norm_budget)
        
        for grid in tqdm(grids):
            weights = {}
            weights.update({stock_code: grid[stock_code] for stock_code in grid})
            
            factor = 1.0 / (sum(weights.values()) + eps)
            weights_norm = {k: v * factor for k, v in weights.items()}
            # current_budget = sum(weights[stock_code] * price[stock_code] for stock_code in weights)
            
            self.portfolio.initialize_portfolio()
            self.portfolio.update_portfolio(weights_norm)
            if self._does_satisfy_constraints():
            # if self._does_satisfy_constraints() and current_budget <= budget:
                candidate_weights.append(weights_norm)
                candidate_objectives.append(self.compute_objective())
                
            else:
            # elif current_budget <= budget:
                non_satisfied_weights.append(weights_norm)
                non_satisfied_objectives.append(self.compute_objective())

        if len(candidate_objectives) != 0:
        
            # Select the best weights
            if self.objective == "cumulative_return":
                best_weights = candidate_weights[np.argmax(candidate_objectives)]
            else:
                best_weights = candidate_weights[np.argmin(candidate_objectives)]

            # Sort the candidates
            final_candidate_weights = [x for _, x in sorted(zip(candidate_objectives, candidate_weights), key=lambda pair: pair[0])]
            final_candidate_objectives = candidate_objectives
            
        else:
            if self.objective == "cumulative_return":
                best_weights = non_satisfied_weights[np.argmax(non_satisfied_objectives)]
            else:
                best_weights = non_satisfied_weights[np.argmin(non_satisfied_objectives)]
            
            # Sort the candidates
            final_candidate_weights = [x for _, x in sorted(zip(non_satisfied_objectives, non_satisfied_weights))]   
            final_candidate_objectives = non_satisfied_objectives
            
        end_time = time.time()
        print(f"Time elapsed: {end_time - start_time} seconds ({(end_time - start_time) / 60} minutes)")
        return best_weights, final_candidate_weights, final_candidate_objectives
    

    def _clustering(self, x, K=10, Niter=10, verbose=True):
        """
        implements Lloyd's algorithm for Cosine similarity metric.
        reference: https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html
        """
        start = time.time()
        N, D = x.shape  # Number of samples, dimension of the ambient space

        c = x[:K, :].clone()  # Simplistic initialization for the centroids
        # Normalize the centroids for the cosine similarity:
        c = torch.nn.functional.normalize(c, dim=1, p=2)

        x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
        c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

        # K-means loop:
        # - x  is the (N, D) point cloud,
        # - cl is the (N,) vector of class labels
        # - c  is the (K, D) cloud of cluster centroids
        for i in range(Niter):
            # E step: assign points to the closest cluster -------------------------
            S_ij = x_i | c_j  # (N, K) symbolic Gram matrix of dot products
            cl = S_ij.argmax(dim=1).long().view(-1)  # Points -> Nearest cluster

            # M step: update the centroids to the normalized cluster average: ------
            # Compute the sum of points per cluster:
            c.zero_()
            c.scatter_add_(0, cl[:, None].repeat(1, D), x)

            # Normalize the centroids, in place:
            c[:] = torch.nn.functional.normalize(c, dim=1, p=2)

        # Select stocks in each cluster with similarity
        cl = cl.cpu().numpy()
        max_index = []
        for i in range(0, cl.max() + 1):
            index = np.where(cl == i)[0]
            current_max = 0
            for j in index.tolist():
                sim = torch.cosine_similarity(x[j], c[i], dim=0)
                if sim > current_max:
                    current_max = sim
                    current_index = j
            max_index.append(current_index)

        if verbose:  # Fancy display -----------------------------------------------
            if use_cuda:
                torch.cuda.synchronize()
            end = time.time()
            print(
                f"K-means for the cosine similarity with {N:,} points in dimension {D:,}, K = {K:,}:"
            )
            print(
                "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                    Niter, end - start, Niter, (end - start) / Niter
                )
            )

        return cl, c, max_index

    
    def _calculate_inertia(self, x, cl, c):
        """Computes the inertia of the K-means clustering given a dataset x,
        a clustering cl and the cluster centroids c.
        """
        x_i = torch.Tensor(x.view(len(x), 1, -1))  # (N, 1, D) samples
        c_j = torch.Tensor(c.view(1, len(c), -1))  # (1, K, D) centroids
        num_indices = torch.arange(len(x)).to(x_i.device)
        cl = torch.LongTensor(cl)  # (N,) labels
        square_dist = ((x_i - c_j) ** 2).sum(dim=-1)
        true_square_dist = square_dist[num_indices, cl]
        sum_of_squared_dist = true_square_dist.sum(dim=-1)
        
        return sum_of_squared_dist
    
    def _compute_stock_embeddings(self):
        stock_embeddings = self.universe.get_stock_embeddings(self.stock_list)
        stock_embeddings = torch.stack(list(stock_embeddings.values())) # (num_stocks, date_length, embedding_dim)
        price_emb = stock_embeddings[:, :, 0] # (num_stocks, date_length)
        # norm_price_emb = torch.nn.functional.normalize(price_emb, dim=1, p=2)
        multi_emb = stock_embeddings.mean(dim=1)[:, 1:] # (num_stocks, embedding_dim)
        stock_embeddings = torch.cat([price_emb, multi_emb], dim=1) # (num_stocks, date_length + embedding_dim)

        # normalize
        stock_embeddings_norm = torch.nn.functional.normalize(stock_embeddings, dim=1, p=2)
        
        return stock_embeddings_norm
    
    def _find_elbow(self):
        
        # Construct stock embeddings
        stock_embeddings_norm = self._compute_stock_embeddings()
        
        # Search inertia per each K, number of cluster 
        sse = []
        for k in range(2, self.num_clusters + 1):
            cl, centroids, _ = self._clustering(stock_embeddings_norm, K=k)
            inertia = self._calculate_inertia(stock_embeddings_norm, cl, centroids)
            sse.append(inertia.sum().item())
            print(f"K = {k}, inertia = {inertia.sum().item()}")

        # Plot the elbow
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(10,8))
        plt.plot(range(2, self.num_clusters + 1), sse)
        plt.xticks(range(2, self.num_clusters + 1))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.savefig(self.args.result_path+f"/{self.args.data_period}_elbow.png")
        plt.close()

        # Find the elbow point
        kl = KneeLocator(range(2, self.num_clusters + 1), sse, curve="convex", direction="decreasing")
        print(f"Elbow point: {kl.knee}")
        
        return kl, sse

    def update_portfolio(self, min_price=None, budget=None, num_stocks=None):

        min_price = self.args.min_price if min_price is None else min_price
        budget = self.args.budget if budget is None else budget

        start_time = time.time()

        best_weights, candidate_weights, candidate_obj = self._grid_search(self.candidates, min_price, budget)

        self.portfolio.update_portfolio(best_weights)
        print("Portfolio updated")
        print(f"Calculated objective: {self.compute_objective()}")

        end_time = time.time()
        print("Heuristic search end ! =============================")
        print(f"Time elapsed: {end_time - start_time} seconds ({(end_time - start_time) / 60} minutes)")

        return best_weights, candidate_weights, candidate_obj
