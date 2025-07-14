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
                    default='lagrange_ours_m')#,] choices=['lagrange_full', 'lagrange_ours',  'lagrange_forward', 'lagrange_backward'])

    parser.add_argument('--cardinality', type=int, default=30)

    # Select the Data to Use
    parser.add_argument('--start_date', type=str, default="2014-01-02")
    parser.add_argument('--end_date', type=str, default="2025-01-02")
    parser.add_argument('--index', type=str,
                    default='sp500')#, choice=['sp500', 'russel, nikkei])


    #nombre de jours 
    parser.add_argument('--T', type=int, default=3, help="nombre d'année pour l'entrainement")
    parser.add_argument('--rebalancing', type=int, default=3, help="Month increment for rebalancing")
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

"""
if __name__ == "__main__":
    Main()
"""



from argparse import Namespace
import matplotlib.pyplot as plt


args = Namespace()
args.index = 'sp500'
args.data_path = 'financial_data'
args.result_path = 'results'
args.solution_name = 'lagrange_ours'
args.rebalancing = 3
args.cardinality = 30
args.start_date = '2014-01-02'
args.end_date = '2025-01-02'

# Initialiser l'objet Universe
universe = Universe(args)
portfolio = Portfolio(universe)
weights = portfolio.rebalance_portfolio(pd.Timestamp("2011-01-02"), pd.Timestamp("2014-01-02"))

test_start  = pd.Timestamp("2014-01-03")
test_end = pd.Timestamp("2015-01-03")

# Extraire rendements entre deux rebalancements
universe.new_universe(test_start, test_end, training=False)
X_test = universe.get_stocks_returns()
Y_test = universe.get_index_returns()


rendements_portefeuille = list(X_test @ weights)
rendements_indice = list(Y_test)
index_dates = list(X_test.index)

# Construire les séries temporelles
rendements_portefeuille = pd.Series(rendements_portefeuille, index=index_dates)
rendements_indice = pd.Series(rendements_indice, index=index_dates)

# Cumuler les rendements
rendements_cumules_portefeuille = (rendements_portefeuille + 1).cumprod() -1
rendements_cumules_indice = (rendements_indice + 1).cumprod() - 1

# Tracer
plt.figure(figsize=(10, 6))  # Taille du graphique

# Tracer les deux courbes
plt.plot(rendements_cumules_portefeuille, label="Portefeuille")
plt.plot(rendements_cumules_indice, label="Indice de référence")

# Ajouter des titres et des labels
plt.title("Performance cumulée (hors-échantillon)")
plt.xlabel("Date")
plt.ylabel("Rendement cumulé")

# Afficher la légende et la grille
plt.legend()

# Afficher le graphique
plt.show()

print(weights)