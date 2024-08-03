import yfinance as yf
import pandas as pd

start_date = "2018-01-01"
end_date = "2023-12-31"
start_year = start_date[:4]
end_year = end_date[:4]

#! change this
index_type = "s&p100" # s&p100, sp500, nasdaq100
#! change This
ticker = "^NDX"  # S&P 100 : ^OEX, nasdaq 100 : ^NDX, S&P 500 : ^GSPC

data_path = '../../NCSOFT/financial_data/'

# 데이터 다운로드
index_data = yf.download(ticker, start=start_date, end=end_date)

# 종가지수 확인
adj_close = index_data['Adj Close']

# 데이터 확인
print(adj_close.head())

# 데이터 저장 (선택 사항)
adj_close.to_csv(f"{data_path}{start_year}_{end_year}_index_{index_type}.csv")
