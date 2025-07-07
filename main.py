import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import os
from prafa.portfolio import Portfolio
from prafa.universe import Universe


def Main():
    # Set the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, 
                    default='financial_data')

    parser.add_argument('--result_path', type=str, 
                    default='results') # default=os.getcwd()+'/results'
    parser.add_argument('--solution_name', type=str,
                    default='lagrange_ours')#,] choices=['lagrange_full', 'lagrange_ours',  'lagrange_forward', 'lagrange_backward'])

    parser.add_argument('--cardinality', type=int, default=30)

    # Select the Data to Use
    parser.add_argument('--start_date', type=str, default="2014-01-02")
    parser.add_argument('--end_date', type=str, default="2025-01-02")
    parser.add_argument('--index', type=str,
                    default='sp500')#, choice=['sp500', 'russel, nikkei])


    #nombre de jours 
    parser.add_argument('--T', type=int, default=2, help="nombre d'année pour l'entrainement")
    parser.add_argument('--rebalancing', type=int, default=6, help="Month increment for rebalancing")
    args = parser.parse_args()
    
  

    
    #fenetre d'entrainement
    portfolio_duration = relativedelta(years=args.T)

    #pour le rebalancement
    time_increment = relativedelta(months=args.rebalancing)

    #liste des dates de rebalancement
    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)
    
    # Construire la liste des dates
    dates = [start_date]
    current_date = start_date + time_increment

    while current_date < end_date:
        dates.append(current_date)
        current_date += time_increment
    
    #initialisation des object necessaire pour extraire les portefeuilles dans le temps
    portfolio = Portfolio(Universe(args))
    for rebalancing_date in dates:
        start_datetime = rebalancing_date - portfolio_duration
        portfolio.rebalance_portfolio(start_datetime, rebalancing_date)
        print(f"Rebalancing from {start_datetime.date()} to {rebalancing_date.date()}")

    
    return None


if __name__ == "__main__":
    Main()


"""

import pickle
from argparse import Namespace
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import pandas as pd
from datetime import datetime


filepath = 'results/portfolio_sp500_lagrange_full_30.json'
with open(filepath, 'rb') as f:
    portfolios = pickle.load(f)


args = Namespace()
args.index = 'sp500'
args.data_path = 'financial_data'
args.result_path = 'results'
args.solution_name = 'lagrange_full'
args.rebalancing = 6  
args.start_date = '2014-01-02'
args.end_date = '2025-01-02'

# Initialiser l'objet Universe
universe = Universe(args)


# S'assurer que les dates sont ordonnées
dates = sorted(list(portfolios.keys()))
n = len(dates)

rendements_portefeuille = []
rendements_indice = []
index_dates = []

for i in range(n - 1):  # On va jusqu'à n-1 car on a besoin de t et t+1
    start_date = dates[i]
    end_date = dates[i + 1] - pd.tseries.offsets.BDay(1)  # veille du prochain rebalance

    # Extraire rendements entre deux rebalancements
    universe.new_universe(start_date, end_date, training=False)
    window_returns = universe.get_stocks_returns()
    weights = list(portfolios[start_date].values())

    # Produit matriciel : rendements du portefeuille
    rp = (window_returns @ weights)
    ri = universe.get_index_returns()

    rendements_portefeuille += rp
    rendements_indice += ri
    index_dates += list(window_returns.index)

# Construire les séries temporelles
rendements_portefeuille = pd.Series(rendements_portefeuille, index=index_dates)
rendements_indice = pd.Series(rendements_indice, index=index_dates)

# Cumuler les rendements
rendements_cumules_portefeuille = (rendements_portefeuille + 1).cumprod() -1
rendements_cumules_indice = (rendements_indice + 1).cumprod() - 1

# Tracer
ax = rendements_cumules_portefeuille.plot(label="Portefeuille")
rendements_cumules_indice.plot(ax=ax, label="Indice de référence", linestyle='--')
ax.set_title("Performance cumulée (hors-échantillon)")
ax.set_ylabel("Rendement cumulé")
ax.legend()
plt.grid(True)
plt.show()
"""