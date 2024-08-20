import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import cvxpy as cp
from datetime import datetime
from pypfopt import expected_returns
from dateutil.relativedelta import relativedelta 
import time
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve
from scipy.optimize import minimize

from prafe.solution.lagrange_orig import Solution
from prafe.solution.snn import SNN
from prafe.utils import *

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True, precision=4)

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
                        default='../NCSOFT/financial_data')
    # parser.add_argument('--data_path', type=str, 
    #                     default=os.getcwd()+'/data_financial')
    parser.add_argument('--result_path', type=str, 
                        default=os.getcwd()+'/results')
    parser.add_argument('--solution_name', type=str,
                        default='lagrange_ours', choices=['lagrange_full', 'lagrange_ours', 'lagrange_ours2', 'lagrange_forward', 'lagrange_backward', 'QP_full', 'QP_forward', 'QP_backward', 'SNN'])
    parser.add_argument('--cardinality', type=int, default=None)
    parser.add_argument('--method', type=str)
    
    # Select the Data to Use
    parser.add_argument('--start_date', type=str, help="2018-01-02", default="2018-01-02")
    parser.add_argument('--end_date', type=str, help="2019-12-31", default="2019-12-31")
    parser.add_argument('--index_type', type=str,
                        default='kospi100')#, choice=['kospi100', 'kosdaq150', 's&p500', 's&p100])

    # Backtesting
    parser.add_argument('--backtesting', type=str2bool, default=False, help="If you want to get daily, weekly, monthly portfolio, set True")
    parser.add_argument('--day_increment', type=int, default=None, help="Day increment")
    parser.add_argument('--month_increment', type=int, default=None, help="Month increment")

    args = parser.parse_args()

    # price, return, index data
    df_price, df_return, df_index, start_date, end_date, start_year, end_year = read_data(args)
    K = args.cardinality
    
    # index type
    index_type = args.index_type
    if index_type == "kospi100":
        # df_index = df_index['IKS100'].pct_change().fillna(value=0.0).apply(lambda x: x if x > -0.999999 else -0.999999)
        # df_index = np.log(1 + df_index)
        df_index = df_index['IKS100'].pct_change().iloc[1:].fillna(value=0.0)
        # df_index = df_index['IKS100'].fillna(value=0.0)
        index_stocks_list = json.load(open(args.data_path + '/stock_list.json'))[index_type][args.start_date]
    elif index_type == "s&p500" or index_type == "s&p100" or index_type == "nasdaq100":
        df_index = df_index['Adj Close'].pct_change().iloc[1:].fillna(value=0.0)
        # df_index = df_index['SPI@SPX'].fillna(value=0.0)
        index_stocks_list = df_price.dropna(axis=1).columns.tolist()
    elif index_type == "kosdaq150":
        # df_index = df_index['IKQ150'].pct_change().fillna(value=0.0).apply(lambda x: x if x > -0.999999 else -0.999999)
        # df_index = np.log(1 + df_index)
        df_index = df_index['IKQ150'].pct_change().iloc[1:].fillna(value=0.0)
        # df_index = df_index['IKQ150'].fillna(value=0.0)
        index_stocks_list = json.load(open(args.data_path + '/stock_list.json'))[index_type][args.start_date]
    # print(df_index[:3])

    # print(index_stocks_list)
    
    
    # Get the universe
    universe = Universe(args = args, df_price= df_price, df_return=df_return, df_index=df_index)

    universe = universe.get_trimmed_universe_by_stocks(list_of_stock_codes=index_stocks_list)
    # universe = universe.get_trimmed_universe_by_time(start_datetime=start_date, end_datetime=end_date)
    
    print(universe._get_universe_datetime_info()) # print the universe datetime info

    # For Rebalancing
    if args.month_increment is not None:
        time_increment = relativedelta(months=args.month_increment)
    elif args.day_increment is not None:
        time_increment = datetime.timedelta(days=args.day_increment)
    else:
        time_increment = datetime.timedelta(days=1)
        
    portfolio_duration = relativedelta(years=1)
    print(f"This is {args.index_type} {args.solution_name} results with Cardinality {args.cardinality}")
    idx = 0
    tracking_errors = []
    tracking_indices = []
    target_indices = []
    start_date_list = []
    end_date_list = []
    current_date = start_date
    current_to_end = current_date + portfolio_duration - pd.Timedelta(days=1)
    rebalancing_date = current_date + time_increment
    print(f"rebalacing_date : {rebalancing_date}")
    rebalancing = "Start"
    # backtesting
    while current_to_end <= end_date or args.backtesting is False: #! backtesting false는 뭐징
        print(f"rebalancing date: {rebalancing_date}")
        print("current_to_end:", current_to_end)
        print("end_date:", end_date)
        
        if args.backtesting:
            # Ignore the date not exists.
            if not universe.is_valid_date(rebalancing_date):
                rebalancing_date += pd.Timedelta(days=1)
                continue
            if not universe.is_valid_date(current_date):
                current_date += pd.Timedelta(days=1)
                current_to_end = current_date + portfolio_duration - pd.Timedelta(days=1)
                continue
            new_universe = universe.get_trimmed_universe_by_time(start_datetime=current_date, end_datetime=current_to_end)
        else:
            new_universe = universe

        
        new_portfolio = Portfolio(new_universe)
        # Define Solution
        if args.solution_name == 'SNN':
            solution = SNN(new_universe, new_portfolio, args.solution_name, args.method, len(index_stocks_list), K)
        else: 
            solution = Solution(new_universe, new_portfolio, args.solution_name, args.method, len(index_stocks_list), K)
        
        ## Update portfolio
        print()
        print("updating portfolio...")
        start_time = time.time()

        if rebalancing == "Start" or rebalancing == True :
            # Rebalancing
            print()
            print(f"{current_date} --> Rebalancing weight!")
            print()
            weights, tracking_error = solution.update_portfolio()
            rebalancing = False
            new_portfolio.update_portfolio(weights) #! 추가
        else:
            # Not Rebalancing
            new_portfolio.update_portfolio(weights)
        
        print("{}-th".format(idx+1))
        print(current_date)

        # Print portfolio
        print_portfolio(weights, K)

        # Evaluate the portfolio
        my_evaluator = Evaluator(universe=new_universe, portfolio=new_portfolio)
        print("====================================")
        print("evaluating portfolio...")
        print_result(my_evaluator)
        print("====================================")

        # Constraint Satisfaction
        print()
        print("checking constraint satisfaction...")
        print_constraints_satisfication(args, new_portfolio, new_universe)
        print("====================================")
        
        # Inference time
        inference_time_sec = time.time() - start_time
        inference_time_min = (time.time() - start_time)/60
        print("inference time: ", inference_time_sec, "seconds (", inference_time_min, "minutes)")
        print("Done!")

        # specify the store path
        if K is not None : 
            store_path = os.path.join(args.result_path, f'{args.index_type}_{args.start_date}_{args.end_date}', str(K), f'{args.solution_name}_{args.method}')
        else:
            store_path = os.path.join(args.result_path, f'{args.index_type}_{args.start_date}_{args.end_date}', f'{args.solution_name}_{args.method}')
        os.makedirs(store_path, exist_ok=True)
        
        formatted_start_date = current_date.strftime('%Y-%m-%d')
        formatted_end_date = current_to_end.strftime('%Y-%m-%d')
        # Save the portfolio
        if args.backtesting:
            new_portfolio.save_portfolio(store_path+f'/{idx+1}th_portfolio_{formatted_start_date}_{formatted_end_date}.csv')
            save_portfolio_csv(store_path + f'/{idx+1}th_evaulation_{formatted_start_date}_{formatted_end_date}.csv', args, new_portfolio, new_universe, my_evaluator, weights, inference_time_sec, inference_time_min)
        else:
            new_portfolio.save_portfolio(store_path+f'/portfolio.csv')
            save_portfolio_csv(store_path + f'/evaulation.csv', args, new_portfolio, new_universe, my_evaluator, weights, inference_time_sec, inference_time_min)

        # Save the Single Stock Solution visualization
        # single_stock_visualization(store_path, idx, new_universe, new_portfolio, my_evaluator, args)
        
        
        # For Tracking Graph
        weight = list(weights.values())
        #! here
        # tracking_index = (new_universe.df_return @ weight).cumprod()[-1]
        tracking_index = (new_universe.df_return @ weight)#.cumsum()[-1]
        tracking_error = new_universe.df_return @ weight - new_universe.df_index
        tracking_errors.append(np.sum(tracking_error**2))
        tracking_indices.append(tracking_index)
        # target_indices.append(new_universe.df_index.cumprod()[-1])
        target_indices.append((new_universe.df_index))#.cumsum())[-1]
        start_date_list.append(formatted_start_date)
        end_date_list.append(formatted_end_date)
        
        print(tracking_index)
        
        if not args.backtesting: 
            break
        
        
        current_date += pd.Timedelta(days=1)
        current_to_end = current_date + portfolio_duration - pd.Timedelta(days=1)
        idx += 1
        
        
        # If rebalancing day, then ** Define new rebalancing date & Define rebalancing as True **
        if current_date >= rebalancing_date:
            print(f"rebalancing date: {rebalancing_date}")
            print("IN!!!!!!!!!!!!!!!!!!!!!!!!!")
            rebalancing_date += time_increment
            rebalancing = True
    
    import pickle

    # pickle 파일로 리스트 저장
    with open(f'./backtesting_' + args.result_path + f'/start_date_list_{args.index_type}.pkl', 'wb') as f:
        pickle.dump(start_date_list, f)
    with open(f'./backtesting_' + args.result_path + f'/end_date_list_{args.index_type}.pkl', 'wb') as f:
        pickle.dump(end_date_list, f)
    with open(f'./backtesting_' + args.result_path + f'/tracking_indices_{args.index_type}_{args.solution_name}_{args.cardinality}.pkl', 'wb') as f:
        pickle.dump(tracking_indices, f)
    with open(f'./backtesting_' + args.result_path + f'/target_indices_{args.index_type}.pkl', 'wb') as f:
        pickle.dump(target_indices, f)
    with open(f'./backtesting_' + args.result_path + f'/tracking_errors_{args.index_type}_{args.solution_name}_{args.cardinality}.pkl', 'wb') as f:
        pickle.dump(tracking_errors, f)
    
    
if __name__ == "__main__":
    main()
    