import json

from IPython.display import display, clear_output
import pandas as pd
from ib_insync import *
import primary_model
import data_preparation
import redis

redis_conn = redis.StrictRedis(host='localhost', port=6379, db=0)


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
        redis_conn.publish('trade_updates', json.dumps(update_data))

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
        durationStr='1 D',
        barSizeSetting='1 min',
        whatToShow='MIDPOINT',
        useRTH=True,
        keepUpToDate=True,
    )

    data.updateEvent += on_new_bar
    ib.run()
























# from forexconnect import ForexConnect, fxcorepy
#
#
# def session_status_changed(session: fxcorepy.O2GSession, status: fxcorepy.AO2GSessionStatus.O2GSessionStatus):
#     print("Trading session status: " + str(status))
#
#
# with ForexConnect() as fx:
#     try:
#         fx.login("D261370901", "Zx6pu", "www.fxcorporate.com/Hosts.jsp", 'Demo', "connection", session_status_callback=session_status_changed)
#
#         # Retrieve the account ID
#         accounts_table = fx.get_table(ForexConnect.ACCOUNTS)
#         account_id = accounts_table.get_row(0, "AccountID").value
#
#         # Retrieve the offer ID for EUR/USD
#         offers_table = fx.get_table(ForexConnect.OFFERS)
#         print(offers_table)
#         for i in range(offers_table.rows_count):
#             if offers_table.row(i, "Instrument").value == "EUR/USD":
#                 offer_id = offers_table.row(i, "OfferID").value
#                 break
#
#         # Additional code for creating and submitting the order here.
#
#     except Exception as e:
#         print("Exception: " + str(e))
#
#     try:
#         fx.logout()
#     except Exception as e:
#         print("Exception: " + str(e))
#

