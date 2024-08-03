import pandas as pd
import os
import json
import datetime 
import pickle
from prafe.universe import Universe
from prafe.portfolio import Portfolio
from prafe.strategy import GenerativeStrategy
from prafe.evaluation import Evaluator
from prafe.constraint import *
from prafe.utils import *

from pathlib import Path
import argparse
import time
import matplotlib.pyplot as plt
import logging

def main():
    
    # Set the logger
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(filename='log.txt')
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=logging.DEBUG)

    # Set the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, 
                        default=os.getcwd()+'/data_financial')
    parser.add_argument('--result_path', type=str, 
                        default=os.getcwd()+'/results')
    parser.add_argument('--portfolio_path', type=str, required=True, help="The file should be pickle file")
    parser.add_argument('--start_date', type=str)
    parser.add_argument('--end_date', type=str)
    parser.add_argument('--objective', type=str, 
                        default='cumulative_return', choices=['cumulative_return', 'variance', 'mdd', 'mdd_duration'])
    parser.add_argument('--solution_name', type=str,
                        default='lagrange', choices=['heuristic_search', 'lagrange', 'mvo', 'reinforcement_learning', 'genetic_algorithm']) 
                        # default='heuristic_search', choices=['heuristic_search', 'lagrange', 'mvo', 'reinforcement_learning', 'genetic_algorithm'])
    parser.add_argument('--strategy_name', type=str)
    
    # Select the Data to Use
    parser.add_argument('--data_period', type=str, 
                        default='2018_2019')#, choice=['2018_2019', '2020_2021_2022'])
    parser.add_argument('--stock_type', type=str,
                        default='kospi')#, choice=['kospi', 'kosdaq'])
    
    def parse_min_stock_ratio(value):
        try:
            return [float(item) for item in value.split(',')]
        except ValueError:
            raise argparse.ArgumentTypeError("min_stock_ratio should be a list of integers")

    # Costraints
    parser.add_argument('--max_weights_sum', type=float, default=1.0)
    parser.add_argument('--max_variance', type=float, default=None)
    parser.add_argument('--max_mdd', type=float, default=None)
    parser.add_argument('--max_mdd_duration', type=float, default=None)
    parser.add_argument('--min_cumulative_return', type=float, default=None)
    parser.add_argument('--min_industry_ratio', nargs='+', default=None, help="'Finance',0.1") #!
    # parser.add_argument('--min_industry_ratio', nargs='+', type=argparse.**arg**parse, help='Specify industry and ratio.')
    parser.add_argument('--min_stock_ratio', type=parse_min_stock_ratio, default=None, help="10,0.1")  # [K, alpha] : 최소 K종목 alpha 이상
    parser.add_argument('--stocks_to_be_included', default=None, nargs='+', help="'A005930' 'A002083'")
    parser.add_argument('--max_stocks_number_constraint', type=int, default=None) 

    # Heuristic Search
    parser.add_argument('--num_clusters', type=int, default=10)
    parser.add_argument('--min_price', type=float, default=1000000.0)  # 1,000,000 KRW
    parser.add_argument('--budget', type=float, default=10000000.0)   # 10,000,000 KRW

    args = parser.parse_args()
    print(args)
    # Get the data
    start_year = ""
    try:
        if args.data_period == "2018_2019":
            df_price = pd.read_pickle(args.data_path + '/processed_2018_2019/price_data.pkl').fillna(value=0.0)
            df_return = pd.read_pickle(args.data_path + '/processed_2018_2019/returns_data.pkl').fillna(value=0.0)
            df_multifactor = pd.read_pickle(args.data_path + '/processed_2018_2019/multifactor_data.pkl').fillna(value=0.0)
            start_year = "2018"
        elif args.data_period == "2020_2021_2022":
            df_price = pd.read_pickle(args.data_path + '/processed_2020_2021_2022/price_data.pkl').fillna(value=0.0)
            df_return = pd.read_pickle(args.data_path + '/processed_2020_2021_2022/returns_data.pkl').fillna(value=0.0)
            df_multifactor = pd.read_pickle(args.data_path + '/processed_2020_2021_2022/multifactor_data.pkl').fillna(value=0.0)
            start_year = "2020"
    except Exception as e:
        print(e)
        df_price, df_return, df_multifactor = get_data_preprocessed(args)
        start_year = args.data_period.split("_")[0]

    # Get the universe
    universe = Universe(args = args, df_price= df_price, df_return=df_return, df_multifactor = df_multifactor)
    time_infos = universe._get_universe_datetime_info()
    print(time_infos)
    
    # Get the example stock list
    if args.stock_type == "kospi":
        trimmed_stock_list = json.load(open(args.data_path + '/stock_list.json'))['코스피100'][start_year+'-01-02']
    elif args.stock_type == "kosdaq":
        trimmed_stock_list = json.load(open(args.data_path + '/stock_list.json'))['코스닥150'][start_year+'-01-02']
    
    trimmed_universe = universe.get_trimmed_universe_by_stocks(list_of_stock_codes=trimmed_stock_list)

    # Make a Portfolio for kospi or kosdaq etc.
    trimmed_portfolio = Portfolio(trimmed_universe)
    print("Initial stock codes of kospi: ", trimmed_portfolio.get_stock_codes())
    print("Initial rewards of kospi: ", trimmed_portfolio.get_rewards())

    # Define Strategy
    strategy = GenerativeStrategy(args, trimmed_portfolio, trimmed_universe)
    original_investments = trimmed_portfolio.investments
    
    with open(args.portfolio_path, 'rb') as handle:
        retrieved_investments = pickle.load(handle)
        
    assert set(original_investments) == set(retrieved_investments), "Stock codes is not equal in two portfolio"
    trimmed_portfolio.update_portfolio(retrieved_investments)

    # Print portfolio
    print_portfolio(retrieved_investments)

    # Evaluate the portfolio
    print("====================================")
    print("evaluating portfolio...")
    my_evaluator = Evaluator(universe = trimmed_universe, portfolio = trimmed_portfolio)
    print("variance         : {:.4f}".format(my_evaluator.calculate_variance()))
    print("AV               : {:.4f}".format(my_evaluator.calculate_AV()))
    print("AAR              : {:.4f}".format(my_evaluator.calculate_AAR()))
    print("CAGR             : {:.4f}".format(my_evaluator.calculate_CAGR()))
    print("cumulative_return: {:.4f}".format(my_evaluator.calculate_cumulative_return()))
    print("Expected_Shortfall: {:.4f}".format(my_evaluator.calculate_Expected_Shortfall()))
    print("Information_Ratio: {:.4f}".format(my_evaluator.calculate_Information_Ratio()))
    print("LPM              : {:.4f}".format(my_evaluator.calculate_LPM()))
    # print("mdd              : {:.4f}".format(my_evaluator.calculate_mdd()))
    # print("mdd_duration     : {:.4f}".format(my_evaluator.calculate_mdd_duration()))
    print("sharpe_ratio     : {:.4f}".format(my_evaluator.calculate_sharpe_ratio()))
    print("calculate_VaR    : {:.4f}".format(my_evaluator.calculate_VaR()))
    print("====================================")

    # Constraint Satisfaction
    print()
    print("checking constraint satisfaction...")

    print_constraints_satisfication(args, trimmed_portfolio, trimmed_universe, strategy)

    print("====================================")
    inference_time_sec = 0
    inference_time_min = 0

    store_path = os.path.join(args.result_path, 'evaluation_{}_{}_{}'.format(args.data_period, args.stock_type, args.strategy_name))
    os.makedirs(store_path, exist_ok=True)
    
    # Save the portfolio
    print(trimmed_portfolio.investments)
    save_portfolio_csv(args.result_path + '/evaluation_{}_{}_{}/result.csv'.format(args.data_period, args.stock_type, args.strategy_name), args, trimmed_portfolio, trimmed_universe, strategy, my_evaluator, retrieved_investments, inference_time_sec, inference_time_min)

    ## Save the Single Stock Solution visualization
    single_stock_visualization(args.result_path + '/evaluation_{}_{}_{}'.format(args.data_period, args.stock_type, args.strategy_name), trimmed_universe, trimmed_portfolio, my_evaluator, args)
    
if __name__ == "__main__":
    main()