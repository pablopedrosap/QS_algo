
import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
# API_TOKEN = '6c6934297be4fd656952161d678cc378846565e8'
#
# headers = {
#     'Content-Type': 'application/json',
#     'Authorization': f'Token {API_TOKEN}'
# }
#
# tickers = ["eurusd", "usdjpy", 'gbpjpy', 'gpbusd', 'usdcad', 'nzdusd']
# dataframes = {}
# for ticker in tickers:
#     dataframes[ticker] = pd.DataFrame(columns=['Price'])
#
# try:
#     while True:
#         url = f"https://api.tiingo.com/tiingo/fx/top?tickers={','.join(tickers)}"
#         response = requests.get(url, headers=headers)
#         data = response.json()
#
#         for entry in data:
#             date = pd.to_datetime(entry["quoteTimestamp"])
#             mid_price = entry["midPrice"]
#             dataframes[entry["index"]].loc[date] = [mid_price]
#
#         # Every iteration, resample and save to CSV
#         for ticker, df in dataframes.items():
#             if not df.empty:
#                 ohlc = df['Price'].resample('1T').ohlc()
#                 ohlc.columns = ["Open", "High", "Low", "Close"]
#                 ohlc_reset = ohlc.reset_index()
#                 ohlc_reset.rename(columns={"index": "Date"}, inplace=True)
#                 ohlc_reset.to_csv(f"{ticker}.csv", index=False)
#
#         time.sleep(5)
#
# except KeyboardInterrupt:
#     print("Interrupted.")

import requests
import csv
from datetime import datetime, timedelta

forex_pairs = [
    'eurusd', 'usdjpy', 'gbpusd', 'usdcad', 'usdchf', 'nzdusd', 'gbpjpy',
    'audusd', 'usdaud', 'euraud', 'eurjpy', 'eurcad',
]

famous_stock_tickers = [
    'aapl', 'msft', 'amzn', 'goog', 'brk.b', 'baba', 'jnj', 'jpm', 'v',
    'pg', 'unh', 'ma', 't', 'hd', 'intc', 'vz', 'tcehy', 'xom', 'wmt',
    'rds.a', 'dis', 'ko', 'bhp', 'cvx', 'pfe', 'mrk', 'pep', 'nvs', 'ba',
    'c', 'tm', 'nke', 'mcd', 'adbe', 'abbv', 'crm', 'nvda', 'bmy', 'cmcsa',
    'bud', 'pypl', 'csco', 'sap', 'orcl', 'tmo', 'abb', 'ibm', 'ul', 'acn',
    'lmt', 'txn', 'nvo', 'unp', 'sbux', 'mmm', 'mdt', 'ge', 'f', 'hdb', 'gild',
    'qcom', 'cost', 'amgn', 'chtr', 'axp', 'amov', 'ne', 'avgo', 'spgi', 'foxa',
    'gd', 'cat', 'dhr', 'gs', 'lly', 'blkb', 'asml', 'adp', 'csx', 'so', 'schw'
]
famous_crypto_tickers = [
    'btc', 'eth', 'bnb', 'ada', 'usdt', 'sol', 'xrp', 'dot', 'doge', 'usdc',
    'avax', 'uni', 'busd', 'link', 'ltc', 'luna', 'matic', 'wbtc', 'atom', 'etc',
    'fil', 'vet', 'bch', 'algo', 'icp', 'xlm', 'trx', 'eos', 'aave', 'xtz',
    'shib', 'theta', 'dai', 'ftt', 'neo', 'hbar', 'ceth', 'celsius', 'cake', 'xmr',
    'amp', 'klay', 'btt', 'iost', 'waves', 'sushi', 'yfi', 'comp', 'snx', 'bat',
    'ar', 'qtum', 'mana', 'uma', 'sand', 'ksm', 'runa', 'near', 'chz', 'stx',
    'zec', 'icx', 'sc', 'enj', 'ren', 'omg', 'rev', 'one', 'iotx', 'hnt', 'yfii',
    'ankr', 'hot', 'nexo', 'pax', 'rune', 'dcr', 'crv', '1inch', 'btg', 'lpt'
]

market_indices = [
    'S&P 500', 'Dow Jones Industrial Average', 'Nasdaq Composite', 'FTSE 100', 'Nikkei 225',
    'Shanghai Composite', 'DAX', 'CAC 40', 'ASX 200', 'Hang Seng Index'
]

API_TOKEN = '6c6934297be4fd656952161d678cc378846565e8'
headers = {'Content-Type': 'application/json'}
url_fx_base = "https://api.tiingo.com/tiingo/fx/audusd/prices?resampleFreq=1min&token=" + API_TOKEN
url_crypto_base = "https://api.tiingo.com/tiingo/crypto/prices?tickers=<>resampleFreq=1min&token=" + API_TOKEN
url_iex_base = "https://api.tiingo.com/iex/<ticker>/prices?startDate=2019-01-02&resampleFreq=5min&token=" + API_TOKEN

start_date = datetime.strptime('2020-01-01', '%Y-%m-%d')
end_date = datetime.strptime('2023-06-01', '%Y-%m-%d')
delta = timedelta(days=3)

all_data = []

for pair in forex_pairs:
    while start_date < end_date:
        url = f"https://api.tiingo.com/tiingo/fx/{pair}/prices?resampleFreq=1min&token={API_TOKEN}&startDate={start_date.strftime('%Y-%m-%d')}&endDate={(start_date + delta).strftime('%Y-%m-%d')}"
        response = requests.get(url, headers=headers)
        data = response.json()
        if data:
            all_data.extend(data)
        start_date += delta + timedelta(minutes=1)  # Increment the date range

    if all_data:
        keys = all_data[0].keys()
        with open('downloads/audusd.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_data)
        print('Data written')
    else:
        print('No data retrieved.')


