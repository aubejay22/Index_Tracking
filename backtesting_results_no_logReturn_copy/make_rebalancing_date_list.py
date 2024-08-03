from dateutil.relativedelta import relativedelta 
from datetime import datetime
import pickle
import pandas as pd


index_type = "kospi100" # kospi100, s&p100, s&p500, nasdaq100

kospi = ["2018-01-02","2018-04-03","2018-07-03","2018-10-02","2019-01-03","2019-04-02","2019-07-02","2019-10-02","2020-01-03","2020-04-02","2020-07-02","2020-10-06","2021-01-05","2021-04-02","2021-07-02","2021-10-05","2022-01-04","2022-04-04"]
sp100 = ["2018-01-03","2018-04-03","2018-07-03","2018-10-02","2019-01-04","2019-04-03","2019-07-03","2019-10-03","2020-01-06","2020-04-03","2020-07-07","2020-10-06","2021-01-06","2021-04-06","2021-07-07","2021-10-06","2022-01-06","2022-04-06"]
sp500 = ["2018-01-02","2018-04-03","2018-07-03","2018-10-02","2019-01-03","2019-04-02","2019-07-02","2019-10-02","2020-01-03","2020-04-02","2020-07-02","2020-10-02","2021-01-05","2021-04-05","2021-07-02","2021-10-04","2022-01-04","2022-04-04"]
nasdaq = ["2018-01-03","2018-04-03","2018-07-03","2018-10-02","2019-01-03","2019-04-02","2019-07-02","2019-10-02","2020-01-03","2020-04-02","2020-07-02","2020-10-02","2021-01-05","2021-04-05","2021-07-02","2021-10-04","2022-01-04","2022-04-04"]

if index_type == "kospi100":
    rebalancing_index = kospi
elif index_type == "s&p100":
    rebalancing_index = sp100
elif index_type == "s&p500":
    rebalancing_index = sp500
elif index_type == "nasdaq100":
    rebalancing_index = nasdaq

portfolio_duration = relativedelta(years=1)
rebalancing_date_list = []
for k in rebalancing_index:
    rebalancing_date = datetime.strptime(k, '%Y-%m-%d')
    rebalancing_date_end = rebalancing_date + portfolio_duration - pd.Timedelta(days=1)
    formatted_rebalancing_end_date = rebalancing_date_end.strftime('%Y-%m-%d')
    rebalancing_date_list.append(formatted_rebalancing_end_date)
    
with open(f'./rebalancing_date_list_{index_type}.pkl', 'wb') as f:
    pickle.dump(rebalancing_date_list, f)
