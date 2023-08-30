import json
import requests
from IPython.display import display, clear_output
import pandas as pd
from ib_insync import *
import primary_model
import data_preparation
import redis

# redis_conn = redis.StrictRedis(host='localhost', port=6379, db=0)


def paper(model, params):
    ib = IB()
    ib.connect('127.0.0.1', 4002, clientId=1)
    contract = Forex('EURUSD')
    ib.qualifyContracts(contract)

    def place_order(direction, qty, df, tp, sl):
        bracket_order = ib.bracketOrder(
            direction,
            qty,
            limitPrice=df.close.iloc[-1],
            takeProfitPrice=tp,
            stopLossPrice=sl,
        )

        for ord in bracket_order:
            ib.placeOrder(contract, ord)

        update_data = {
            'status': 'new_order',
            'data': bracket_order
        }
        # redis_conn.publish('trade_updates', json.dumps(update_data))

    def on_new_bar(data, has_new_bar: bool):
        if has_new_bar:
            df = util.df(data)
            print(df)
            print(df.columns)
            df = primary_model.primary_features(df, params, live=True)
            df = data_preparation.secondary_features(df, params)

            def get_side(row, features):
                signals = [row[feature] for feature in features]
                side = 1 if sum(signals) == len(features) else -1 if sum(signals) == -len(features) else 0
                return side

            df['side'] = df.apply(lambda row: get_side(row, params['features']), axis=1)
            print(df)
            print(df.columns)

            if df.side.iloc[-1] == 1:
                place_order('BUY', 1, df, df.close.iloc[-1] + 0.1, df.close.iloc[-1] - 0.1)

            elif df.side.iloc[-1] == -1:
                place_order('SELL', 1, df, df.close.iloc[-1] - 0.1, df.close.iloc[-1] + 0.1)

    data = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='3 D',
        barSizeSetting='5 mins',
        whatToShow='MIDPOINT',
        useRTH=True,
        keepUpToDate=True,
    )

    data.updateEvent += on_new_bar
    ib.run()


# CONSUMER_KEY = 'DICIEMBRE'
# SECRET_TOKEN = 'twxfwJQKa8futHCwcE1EPVNb9RGaMPv/LZcVH8dFOzy3p3l3djOJdE72mRFWktugDH29UbOmG7ceO5kRqmONYj/3Pm0SzqkYbAM9O/GafOyA4DEnomrnyPg0WYG/Bzee7JWc5cNTZzV5lepVnHLcsMlLNc/4AGBjCyUc49aDLAEc0dSdW+5q38gg+m00PTJWLWWYUr7v50XuEpW6zvoiTB6awLi0FU4p3+o3SThwdoQJ2sjNgDnjBKkG4IZ+Gl6MKiR1Ctz6Ho52JeCAevTaZsoIqt+UZtJRdIaO2lifwk0iRQMPmb6L8bALSsnq1D8uB+r/bESN1D2Yjv0NPSUm+A=='
# ACCESS_TOKEN = '6810df8b5b06b041ffea'
#
# import requests
# import time
# import uuid
#
#
# YOUR_SIGNATURE = 'Pablo Pedrosa'
# headers1 = {
#     "Authorization": f"OAuth realm='limited_poa', 'OAuth oauth_consumer_key='DICIEMBRE', oauth_signature_method='HMAC-SHA1', oauth_signature='{YOUR_SIGNATURE}', oauth_timestamp='{int(time.time())}', oauth_nonce='{uuid.uuid4().hex}'"
# }
#
# response1 = requests.post("IB_REQUEST_TOKEN_URL", headers=headers1)
#
# headers3 = {
#     "Authorization": f'OAuth oauth_consumer_key="{CONSUMER_KEY}", oauth_token="{UNAUTHORIZED_REQUEST_TOKEN}", oauth_signature_method="HMAC-SHA1", oauth_signature="{YOUR_SIGNATURE}", oauth_timestamp="{int(time.time())}", oauth_nonce="{uuid.uuid4().hex}"'
# }
#
# response3 = requests.post("IB_ACCESS_TOKEN_URL", headers=headers3)




