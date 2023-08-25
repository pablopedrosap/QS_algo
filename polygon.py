import pandas as pd
import requests
import csv
import json
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def format_date(date):
    return date.strftime('%Y-%m-%d')


def get_data(symbol, start_date, end_date):

    url = f"https://api.polygon.io/v2/aggs/ticker/C:{symbol}/range/1/minute/{start_date}/{end_date}?adjusted=true&sort=asc&limit=50000&apiKey=GP42LGVELIp_JK1LfWBMXbQW5mQia5qE"
    response = requests.get(url)
    print(response)
    time.sleep(15)
    data = json.loads(response.text)

    with open(f'{symbol}_data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        results = data.get('results', [])

        for result in results:
            writer.writerow([
                result['t'],
                result['o'],
                result['h'],
                result['l'],
                result['c'],
                result['v'],
                result.get('n', None),
                result.get('vw', None),
            ])

    if results:

        last_timestamp = results[-1]['t']
        last_date = datetime.utcfromtimestamp(last_timestamp // 1000)
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

        if last_date > end_date:
            return None
        else:
            return format_date(last_date + timedelta(minutes=1))
    else:
        return None


forex_pairs = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']


# end_date_datetime = datetime.strptime(end_date_str, '%Y-%m-%d')

for forex in forex_pairs:
    start_date = datetime(2021, 1, 9)
    end_date = datetime(2023, 8, 1)

    start_date_str = format_date(start_date)
    end_date_str = format_date(end_date)
    with open(f'{forex}_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'No. of Transactions', 'VWAP'])

    while True:
        start_date_str = get_data(forex, start_date_str, end_date_str)
        if start_date_str is None:
            break

        if start_date_str > end_date_str:
            break

