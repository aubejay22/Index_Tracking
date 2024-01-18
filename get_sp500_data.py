import yfinance as yf
from pandas_datareader import DataReader
from datetime import datetime
import pandas as pd

#! Change Year
year = '2018'
data_path = '../NCSOFT/financial_data/'
# sp500 = yf.Ticker('^KS100')

# # get all stock info
# print(sp500.info)

# # get historical market data
# # hist = sp500.history(period='1mo')
# hist = sp500.history(start='2018-01-01', end='2018-12-31')
# # print(sp500)
# print(hist['Close'])
# # print(sp500.history_metadata)


import bs4 as bs
import requests
import yfinance as yf
import datetime

resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table', {'class': 'wikitable sortable'})

tickers = []

for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    tickers.append(ticker)
    
# print(tickers)
tickers = [s.replace('\n', '') for s in tickers]
# print(len(tickers))
# print(tickers)


tickers_list = ""
for ticker in tickers:
    tickers_list += ticker + " "
print(tickers_list)

data = yf.download(tickers_list, start=f'{year}-01-01', end=f'{year}-12-31',interval="1d",)
df = pd.DataFrame(data)
# print(df.columns)
df = df['Close']
print(df.dropna(axis=1).columns.tolist())
# df.to_csv(f'{data_path}SP500_price_data_{year}.csv')
