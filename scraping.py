

from datetime import timedelta, datetime
from io import StringIO

import numpy as np
import pandas as pd
from selenium.webdriver.chrome.service import Service
import csv
import random
from datetime import timedelta, datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

'''
with open('downloads/economic_events.csv', mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    writer.writerow(['Date/Time', 'Currency', 'Event', 'Actual', 'Forecast', 'Previous'])

    start_date = datetime(2022, 12, 26)
    end_date = datetime(2023, 7, 1)

    while start_date <= end_date:
        # Create a new WebDriver session for each loop
        option = webdriver.ChromeOptions()
        option.add_argument("start-maximized")
        option.add_argument('--headless=new')

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=option)

        try:
            month = start_date.strftime('%b')
            day_of_month = start_date.day
            year = start_date.year
            url = f"https://www.forexfactory.com/calendar?day={month}{day_of_month}.{year}"
            driver.get(url)
            time.sleep(random.randint(4, 8))

            table = driver.find_element(By.CLASS_NAME, "calendar__table")

            for row in table.find_elements(By.TAG_NAME, "tr"):
                # List comprehension to get each cell's data and filter out empty cells
                row_data = list(filter(None, [td.text.replace('\n', ' ') for td in row.find_elements(By.TAG_NAME, "td")]))
                try:
                    if any(weekday in row_data[0] for weekday in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']):
                        day = row_data[0]
                        row_data = row_data[1:]

                    if any(char in row_data[0] for char in ['am', 'pm']):
                        hour = row_data[0]
                        row_data = row_data[1:]
                    if day:
                        row_data.insert(0, day)
                    if hour:
                        row_data.insert(1, hour)
                    if len(row_data) != 7:
                        continue
                except:
                    pass

                if len(row_data) != 7:
                    continue
                print(row_data)
                writer.writerow(row_data)

        except Exception as e:
            print(e)

        finally:
            driver.quit()

        rand = random.randint(30, 60)
        print(f"Sleeping for {rand} seconds...")
        time.sleep(rand)

        start_date += timedelta(days=1)

driver.quit()
'''


economic_df = pd.read_csv('/Users/pablopedrosa/PycharmProjects/QS_algorithm/downloads/economic_events.csv')

event_keywords = ['PMI', 'Unemployment', 'Trade Balance', 'Natural Gas', 'Oil Inventories', 'CPI', 'PPI', 'GDP', 'Confidence']
economic_df = economic_df[economic_df['Event'].apply(lambda x: any(keyword in x for keyword in event_keywords))]

interested_currencies = ['EUR', 'USD']
economic_df = economic_df[economic_df['Currency'].isin(interested_currencies)]
economic_df.reset_index(drop=True, inplace=True)
economic_df['Year'] = 2019

for i in range(1, len(economic_df)):
    if economic_df.loc[i, 'Date'].split()[1] == 'Jan' and economic_df.loc[i-1, 'Date'].split()[1] == 'Dec':
        economic_df.loc[i, 'Year'] = economic_df.loc[i-1, 'Year'] + 1
    else:
        economic_df.loc[i, 'Year'] = economic_df.loc[i-1, 'Year']

try:
    economic_df['Datetime'] = pd.to_datetime(economic_df['Year'].astype(str) + ' ' + economic_df['Date'] + ' ' + economic_df['Time'], format='%Y %a %b %d %I:%M%p')
except Exception as e:
    print("Datetime conversion failed:", e)


feature_df = pd.DataFrame(index=economic_df['Datetime'].unique()).copy()
feature_df.index.name = 'Datetime'
feature_df.sort_index(inplace=True)


for index, row in economic_df.iterrows():
    event = row['Event']

    # Calculate 'Actual - Forecast' difference
    try:
        actual = float(row['Actual'].strip('%MmKk'))
        forecast = float(row['Forecast'].strip('%MmKk'))
    except ValueError:
        continue
    diff = actual - forecast

    # Get 'Previous' value
    try:
        previous = float(row['Previous'].strip('%MmKk'))
    except ValueError:
        continue

    feature_df.loc[row['Datetime'], f'{event}_Diff'] = diff
    feature_df.loc[row['Datetime'], f'{event}_Previous'] = previous

pd.set_option('display.max_columns', None)
feature_df = feature_df.resample('T').ffill()
for col in feature_df.filter(like='_Diff').columns:
    feature_df[col] = feature_df[col].fillna(method='ffill')

for col in feature_df.filter(like='_Previous').columns:
    feature_df[col] = feature_df[col].fillna(method='ffill')


feature_df.dropna(inplace=True)
print(feature_df)
print(economic_df)

#
# for keyword in event_keywords:
#     feature_df[f'{keyword}_TimeSinceLastEvent'] = np.nan
#     feature_df[f'{keyword}_TimeUntilNextEvent'] = np.nan
#
# last_event_time_dict = {keyword: None for keyword in event_keywords}
# next_event_time_dict = {keyword: None for keyword in event_keywords}


# for current_time in feature_df.index:
#     # Identify rows in economic_df that have the same datetime as current_time
#     matching_rows = economic_df[economic_df['Datetime'] == current_time]
#
#     for _, row in matching_rows.iterrows():
#         for keyword in event_keywords:
#             if keyword in row['Event']:
#                 last_event_time_dict[keyword] = current_time
#
#     for keyword in event_keywords:
#         feature_df.loc[current_time, f'{keyword}_TimeSinceLastEvent'] = (current_time - last_event_time_dict[
#             keyword]).total_seconds() / 60 if last_event_time_dict[keyword] is not None else np.nan
#
# for current_time in reversed(feature_df.index):
#     matching_rows = economic_df[economic_df['Datetime'] == current_time]
#
#     for _, row in matching_rows.iterrows():
#         for keyword in event_keywords:
#             if keyword in row['Event']:
#                 next_event_time_dict[keyword] = current_time
#
#     for keyword in event_keywords:
#         feature_df.loc[current_time, f'{keyword}_TimeUntilNextEvent'] = (next_event_time_dict[
#                                                                              keyword] - current_time).total_seconds() / 60 if \
#             next_event_time_dict[keyword] is not None else np.nan






