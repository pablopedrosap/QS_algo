

from datetime import timedelta, datetime
from selenium.webdriver.chrome.service import Service
import csv
import random
from datetime import timedelta, datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time


with open('economic_events.csv', mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    writer.writerow(['Date/Time', 'Currency', 'Event', 'Actual', 'Forecast', 'Previous'])

    start_date = datetime(2020, 12, 5)
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
