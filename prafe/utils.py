import pandas as pd
import os
import json
import datetime 
from prafe.universe import Universe
from prafe.portfolio import Portfolio
from prafe.strategy import Strategy, GenerativeStrategy
from prafe.solution import Solution, HeuristicSearch, MVO
from prafe.solution.lagrange import Lagrange
from prafe.evaluation import Evaluator
from prafe.constraint import *

from pathlib import Path
import argparse
import time
import matplotlib.pyplot as plt
import warnings
import logging
import csv


def get_data_preprocessed(args, start_year, end_year):

    data_path = args.data_path

    df_prices = []
    df_returns = []
    df_multifactors = []
    
    current_year = start_year
    for current_year in range(start_year, end_year+1):
        df_prices.append(pd.read_csv(data_path + '/price_data_' + str(current_year) + '.csv', parse_dates=True, index_col="date").fillna(value=0.0))
        df_returns.append(pd.read_csv(data_path + '/returns_data_' + str(current_year) + '.csv', parse_dates=True, index_col="date").fillna(value=0.0))
        df_multifactors.append(pd.read_csv(data_path + '/multifactor_data_' + str(current_year) + '.csv', parse_dates=['date']).fillna(value=0.0))

    df_price = pd.concat(df_prices)
    df_return = pd.concat(df_returns)
    df_multifactor = pd.concat(df_multifactors)
    
    os.makedirs(data_path + '/processed_' + str(start_year) + '_' + str(end_year), exist_ok=True)
    df_price.to_pickle(data_path + '/processed_' + str(start_year) + '_' + str(end_year) + '/price_data.pkl')
    df_return.to_pickle(data_path + '/processed_' + str(start_year) + '_' + str(end_year) + '/return_data.pkl')
    df_multifactor.to_pickle(data_path + '/processed_' + str(start_year) + '_' + str(end_year) + '/multifactor_data.pkl')  

    return df_price, df_return, df_multifactor


def print_constraints_satisfication(args, portfolio, universe, strategy):
    if args.max_weights_sum : 
        print("weights sum                  : ", weights_sum_constraint(portfolio, universe, strategy))
    if args.min_cumulative_return :
        print("cumulative_return_constraint : ", cumulative_return_constraint(portfolio, universe, strategy))
    if args.max_stocks_number_constraint : 
        print("stocks_number_constraint     : ", stocks_number_constraint(portfolio, universe,strategy))
    if args.min_industry_ratio : 
        print("industry_ratio_constraint: ", industry_ratio_constraint(portfolio, universe, strategy))
    if args.min_stock_ratio : 
        print("stock_ratio_constraint       : ", stock_ratio_constraint(portfolio, universe, strategy))
    if args.max_mdd : #!
        print("mdd_constraint               : ", mdd_constraint(portfolio, universe, strategy))
    if args.max_mdd_duration : #!
        print("mdd_duration_constraint      : ", mdd_duration_constraint(portfolio, universe, strategy))
    if args.max_variance :
        print("variance_constraint          : ", variance_constraint(portfolio, universe, strategy))
    if args.stocks_to_be_included :
        print("including_stocks_constraint  : ", including_stocks_constraint(portfolio, universe, strategy))
        
        
def print_portfolio(weights):
    weight_sum = 0
    eps = 1e-4 # 0.0001
    print()
    for stock in weights.keys():
        weight_sum += weights[stock]
        if weights[stock] >= eps :
            print('{} : {:.4f}'.format(stock, weights[stock]))
    print("(weight sum) = {}".format(weight_sum))
    print()
    
    
def save_portfolio_csv(
    path : str,
    args, 
    portfolio : Portfolio, 
    universe : Universe, 
    strategy : GenerativeStrategy, 
    evaluator : Evaluator,
    weights,
    inference_time_sec,
    inference_time_min
    ):
        ## Evaluating Values
        CR = evaluator.calculate_cumulative_return()
        CAGR = evaluator.calculate_CAGR()
        AAR = evaluator.calculate_AAR()
        Variance = evaluator.calculate_variance()
        AV = evaluator.calculate_AV()
        SR = evaluator.calculate_sharpe_ratio()
        LPM = evaluator.calculate_LPM()
        VaR = evaluator.calculate_VaR()
        ES = evaluator.calculate_Expected_Shortfall()
        # IR = evaluator.calculate_Information_Ratio()
        MDD = evaluator.calculate_mdd()
        Max_Drawdown_duration = evaluator.calculate_recovery_time()
        
        # portfolio_evaluation = [
        #     ["Cumulative Return", "CAGR", "AAR", "Variance", "AV", "Sharpe Ratio", "LPM", "VaR", "Expected Shortfall", "Information Ratio", "MDD", "Max Drawdown_Duration"],
        #     [CR, CAGR, AAR, Variance, AV, SR, LPM, VaR, ES, IR, MDD, Max_Drawdown_duration]
        # ]
        #! Except "Information Ratio"
        portfolio_evaluation = [
            ["Cumulative Return", "CAGR", "AAR", "Variance", "AV", "Sharpe Ratio", "LPM", "VaR", "Expected Shortfall", "MDD", "Max Drawdown_Duration"],
            [CR, CAGR, AAR, Variance, AV, SR, LPM, VaR, ES, MDD, Max_Drawdown_duration]
        ]
        
        ## Constraint Satisfication
        satisfications = {}
        satisfications["Weight Sum"] = weights_sum_constraint(portfolio, universe, strategy)
        satisfications["Expected Return constraint"] = cumulative_return_constraint(portfolio, universe, strategy)
        satisfications["Variance constraint"] = variance_constraint(portfolio, universe, strategy)
        satisfications["Stock Number constraint"] = stocks_number_constraint(portfolio, universe,strategy)
        satisfications["Industry Ratio constraint"] = industry_ratio_constraint(portfolio, universe, strategy)
        # if strategy.min_industry_ratio != None:
        #     satisfications["Industry Ratio constraint"] = industry_ratio_constraint(portfolio, universe, strategy)
        # else:
        #     satisfications["Industry Ratio constraint"] = None
        satisfications["Stock Ratio constraint"] = stock_ratio_constraint(portfolio, universe, strategy)
        satisfications["MDD constraint"] = mdd_constraint(portfolio, universe, strategy)
        satisfications["MDD duration constraint"] = mdd_duration_constraint(portfolio, universe, strategy)
        satisfications["Including Stocks constraint"] = including_stocks_constraint(portfolio, universe, strategy)
        
        portfolio_satisfication = [[],[]]
        for constraint, satisfied in satisfications.items():
            portfolio_satisfication[0].append(constraint)
            portfolio_satisfication[1].append(satisfied)
        print(portfolio_satisfication)
        
        ## Inference Time
        inference_time = [["Seconds", "Minutes"], [inference_time_sec, inference_time_min]]
        
        total_result = [[], []]
        total_result[0].extend(portfolio_evaluation[0])
        total_result[0].extend(portfolio_satisfication[0])
        total_result[0].extend(inference_time[0])
        total_result[1].extend(portfolio_evaluation[1])
        total_result[1].extend(portfolio_satisfication[1])
        total_result[1].extend(inference_time[1])
        
        with open(path, 'w', newline="") as f :
            writer = csv.writer(f)
            writer.writerow(total_result[0])
            writer.writerow(['None' if x is None else x for x in total_result[1]])
        
def single_stock_visualization(
        path: str,
        idx: int,
        trimmed_universe: Universe, 
        trimmed_portfolio: Portfolio, 
        my_evaluator: Evaluator,
        args
    ): 
    single_stock_weights = trimmed_portfolio.get_weights_dict()
    returns = {}
    variances = {}
    sharpe_ratios = {}
    
    solution_name = args.solution_name
    returns[solution_name] = my_evaluator.calculate_CAGR()
    variances[solution_name] = my_evaluator.calculate_AV()
    sharpe_ratios[solution_name] = my_evaluator.calculate_sharpe_ratio() 

    for stock in single_stock_weights.keys():
        single_stock_weights[stock] = 1.0
        trimmed_portfolio.update_portfolio(single_stock_weights)
        current_eval = Evaluator(trimmed_universe, trimmed_portfolio)
        returns[stock] = current_eval.calculate_CAGR()
        variances[stock] = current_eval.calculate_AV()
        sharpe_ratios[stock] = current_eval.calculate_sharpe_ratio()
        trimmed_portfolio.initialize_portfolio()
    
    eps = 1e-2
    min_returns, max_returns = min(returns.values())-eps, max(returns.values())
    min_variances, max_variances = min(variances.values())-eps, max(variances.values())
    min_sharpe, max_sharpe = min(sharpe_ratios.values())-eps, max(sharpe_ratios.values())
    scaled_returns = {key: (value - min_returns) / (max_returns - min_returns) for key, value in returns.items()}
    scaled_variances = {key: (value - min_variances) / (max_variances - min_variances) for key, value in variances.items()}
    scaled_sharpe = {key: (value - min_sharpe) / (max_sharpe - min_sharpe) for key, value in sharpe_ratios.items()}

    returns_sorted = dict(sorted(scaled_returns.items(), key=lambda x: x[1], reverse=True))
    var_sorted = dict(sorted(scaled_variances.items(), key=lambda x: x[1]))
    sharpe_sorted = dict(sorted(scaled_sharpe.items(), key=lambda x: x[1], reverse=True))

    os.makedirs(path + '/Single_Portfolio_Visualization', exist_ok=True)
    # Bar chart - Cumulative Return
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(returns_sorted)), list(returns_sorted.values()), color='#D0CECE', align='center')
    plt.bar(list(returns_sorted.keys()).index(solution_name), returns_sorted[solution_name], color='r', align='center')
    # plt.xticks(range(len(returns_sorted)), list(returns_sorted.keys()), rotation=90)
    plt.xlabel('Single Portfolio')
    plt.title('Cumulative Return')
    plt.savefig(path + f"/Single_Portfolio_Visualization/{idx+1}th_CAGR.png")
    plt.close()

    # Bar Chart - Annual Volatility
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(var_sorted)), list(var_sorted.values()), color='#D0CECE', align='center')
    plt.bar(list(var_sorted.keys()).index(solution_name), var_sorted[solution_name], color='r', align='center')
    # plt.xticks(range(len(var_sorted)), list(var_sorted.keys()), rotation=90)
    plt.xlabel('Single Portfolio')
    plt.title('Annual Volatility')
    plt.savefig(path + f"/Single_Portfolio_Visualization/{idx+1}th_AV.png")
    plt.close()
    
    # Bar Chart - Sharpe Ratio
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(sharpe_sorted)), list(sharpe_sorted.values()), color='#D0CECE', align='center')
    plt.bar(list(sharpe_sorted.keys()).index(solution_name), sharpe_sorted[solution_name], color='r', align='center')
    # plt.xticks(range(len(sharpe_sorted)), list(sharpe_sorted.keys()), rotation=90)
    plt.xlabel('Single Portfolio')
    plt.title('Sharpe Ratio')
    plt.savefig(path + f"/Single_Portfolio_Visualization/{idx+1}th_SR.png")
    plt.close()
    
def parse_min_stock_ratio(value):
    try:
        return [float(item) for item in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("min_stock_ratio should be a list of integers")
    
def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ['True', 'true', 'TRUE', 'T', 't']:
        return True
    elif value.lower() in ['False', 'false', 'FALSE', 'F', 'f']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def read_data(args):
    start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(args.end_date, '%Y-%m-%d')
    start_year = start_date.timetuple()[0]
    end_year = end_date.timetuple()[0]
    
    try:
        df_price = pd.read_pickle(args.data_path + '/processed_' + str(start_year) + '_' +str(end_year) + '/price_data.pkl').fillna(value=0.0)
        df_return = pd.read_pickle(args.data_path + '/processed_' + str(start_year) + '_' + str(end_year) + '/return_data.pkl').fillna(value=0.0)
        df_multifactor = pd.read_pickle(args.data_path + '/processed_' + str(start_year) + '_' + str(end_year) + '/multifactor_data.pkl').fillna(value=0.0)
    except Exception as e:
        print(e)    
        df_price, df_return, df_multifactor = get_data_preprocessed(args, start_year, end_year)
    
    return df_price, df_return, df_multifactor, start_date, end_date, start_year, end_year
        

def print_result(my_evaluator):
    print("variance         : {:.4f}".format(my_evaluator.calculate_variance()))
    print("AV               : {:.4f}".format(my_evaluator.calculate_AV()))
    print("AAR              : {:.4f}".format(my_evaluator.calculate_AAR()))
    print("CAGR             : {:.4f}".format(my_evaluator.calculate_CAGR()))
    print("cumulative_return: {:.4f}".format(my_evaluator.calculate_cumulative_return()))
    print("Expected_Shortfall: {:.4f}".format(my_evaluator.calculate_Expected_Shortfall()))
    # print("Information_Ratio: {:.4f}".format(my_evaluator.calculate_Information_Ratio()))
    print("LPM              : {:.4f}".format(my_evaluator.calculate_LPM()))
    print("sharpe_ratio     : {:.4f}".format(my_evaluator.calculate_sharpe_ratio()))
    print("calculate_VaR    : {:.4f}".format(my_evaluator.calculate_VaR()))
    print("MDD              : {:.4f}".format(my_evaluator.calculate_mdd()))
    print("Max Drawdown_duration: {:.4f}".format(my_evaluator.calculate_recovery_time()))
    
def search_solution(
        args: argparse.Namespace,
        trimmed_universe: Universe,
        trimmed_portfolio: Portfolio, 
        strategy: Strategy
    ):
    solution_name = args.solution_name
    solution = None
    if solution_name == 'lagrange':
        solution = Lagrange(trimmed_universe, trimmed_portfolio, strategy)
    elif solution_name == 'heuristic_search':
        solution = HeuristicSearch(args, trimmed_universe, trimmed_portfolio, strategy)
    elif solution_name == 'mvo':
        solution = MVO(trimmed_universe, trimmed_portfolio, strategy)
    else:
        raise Exception("Not yet completed solution type!")

    return solution