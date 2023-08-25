import pickle
import sys
import json
import pandas as pd
import data_preparation
import primary_model
import model_training
import ensemble
import walkforward
import cProfile
import trading
import json
import matplotlib.pyplot as plt
import os
import sys


def dataPreparation(params):
    # if params["trading_frequency"] == 'weekly':
    #     timeframe = 'h1'
    # elif params["trading_frequency"] == 'daily':
    #     timeframe = 'm15'
    # elif params["trading_frequency"] == 'hourly':
    #     timeframe = 'm5'
    # else:
    #     timeframe = 'm1'

    df = data_preparation.csv(f'/Users/pablopedrosa/PycharmProjects/QS_algorithm/downloads/'
                              f'{params["asset_name"][0].replace("/", "")}_data.csv', 'm1')
    df = primary_model.primary_features(df, params)
    df = data_preparation.secondary_features(df, params)
    to_backtest = df

    t_events = data_preparation.cusum(df)
    df, events = data_preparation.add_barriers(df, t_events)
    events = data_preparation.weights(events, df)
    bins = data_preparation.get_bins(events, df['close'])
    clean_bins = data_preparation.drop_labels(bins, minPtc=.05)
    df.reset_index(inplace=True)
    # df = data_preparation.calculate_fracdiff(df)
    df.dropna(inplace=True)
    df.index = df['timestamp']
    to_model = df.join(clean_bins, how='inner')
    to_model.dropna(inplace=True)

    return to_model, to_backtest


def main(params):
    results = {}
    if params['action'] == 'create' or params['action'] == 'update':
        to_model, to_backtest = dataPreparation(params)
        best_model = model_training.testFunc(to_model[:int(len(to_model)*0.3)], params)
        # model_dir = "models"
        # model_filename = f'{params["userID"]}model_{params["features"]}.pkl'
        # model_filepath = os.path.join(model_dir, model_filename)
        # print(model_filepath)
        # with open(model_filepath, 'wb') as model_file:
        #     pickle.dump(best_model, model_file)

        results = walkforward.main(best_model, to_model, to_backtest, params)
        # print(results)

        # sys.stdout.write(json.dumps(results) + '\n')
        # sys.stdout.flush()
        # sys.exit(0)

    elif params['action'] == 'paper' or params['action'] == 'live':
        try:
            model = pickle.load(open(f'/Users/pablopedrosa/PycharmProjects/QS_algorithm/models/'
                                     f'{params["userID"]}model_{params["features"]}.pkl', 'rb'))
        except:
            model = ''

        if params['action'] == 'paper':
            results = trading.paper(model, params)
        else:
            results = trading.live(model, params)
    else:
        raise ValueError(f'Unknown action: {params["action"]}')


if __name__ == '__main__':
    # params = json.loads(sys.argv[1])

    # params = {'userID': 29, 'name': "Forex ['Government Debt and Fiscal Policy']", 'asset_name': ['GBP/USD'],
    #           'fundamental_feature': ['Government Debt and Fiscal Policy'],
    #           'sentiment_feature': ['Market Sentiment Indexes'],
    #           'technical_feature': ['Trendlines', 'Pivot Points'],
    #           'long_entry_signal': {'selectedLongEntrySignals': ['Composition of government spending']},
    #           'short_entry_signal': {'selectedShortEntrySignals': ['Tax policy impacts']}, 'action': 'create'}
    #
    # params['ml_model'] = 'random_forest'
    # params['risk_tolerance'] = 'High'

    params = {
        'userID': '1',
        'name': 'fuuu',
        'description': '',
        'strategy_objective': 'risk_minimization',
        'asset_selection': 'forex',
        'asset_name': ['EURUSD'],
        'trading_frequency': 'minutes', 'position_holding': 'short_term', 'risk_tolerance': 'low',
        'fundamental_feature': 'nope',
        'sentiment_feature': ['news_headlines', 'retail_sentiment'],
        'technical_feature': ['higher_highs_lows'],
        'features': [],
        'entry_long_signal': ['price_above_sma', 'retracement', 'small_break_structure'],

        'exit_long_signal': 'price_crosses_below_sma',
        'entry_short_signal': ['price_above_sma', 'retracement', 'small_break_structure'],
        'exit_short_signal': 'price_crosses_below_sma',
        'initial_investment': 100000,
           'ml_model': 'random_forest',

        'action': 'paper'}
    main(params)


'''NEXT STEPS'''

'''strategy creation flow'''

'''show backtesting in dash'''

'''deploy user model to paper trading'''

'''add features, alphavantage quandl'''

'''add weights, purgedkfold, dynamic bet_sizing, multiprocessing'''




