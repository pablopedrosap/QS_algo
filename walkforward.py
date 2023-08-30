import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from bet_sizing import getSignal
from backtesting import Backtest, Strategy
import bokeh
import time


def get_trade_direction(data, features):
    unique_values = set(data[feature] for feature in features)
    if len(unique_values) == 1 and 0 not in unique_values:
        return unique_values.pop()
    return 0


class MyStrategy(Strategy):
    risk_tolerance = None
    features = None
    secret_features = None
    model = None

    def init(self):
        self.current_tp = None
        self.current_sl = None

    def next(self):
        current_data = pd.Series({
            'index': self.data.index[-1],
            'Open': self.data.Open[-1],
            'High': self.data.High[-1],
            'Low': self.data.Low[-1],
            'Close': self.data.Close[-1],
            'trgt_pips': 0.005,
        })
        for feature_name in self.features:
            if hasattr(self.data, feature_name):
                current_data[feature_name] = getattr(self.data, feature_name)[-1]

        SL_pips = current_data['trgt_pips']
        trade_direction = get_trade_direction(current_data, self.features)

        if trade_direction != 0:
            bet_size = 1#self.model.predict_proba(current_data[self.features + self.secret_features])[0][1]
            risk = {'high': 0.02, 'low': 0.001}.get(self.risk_tolerance, 0.05)
            number_of_lots = (self.equity * risk) / (SL_pips * 10) * bet_size
            position_value = 10000  # Just a placeholder

            # Implement METAMODEL BET SIZING
            if trade_direction == 1:
                self.buy(size=.1, tp=current_data.Close + SL_pips*1, sl=current_data.Close - SL_pips)
            else:
                self.sell(size=.1, tp=current_data.Close - SL_pips*1, sl=current_data.Close + SL_pips)


def walk_forward_backtest(data_model, data_backtest, model, params):
    window_size = round(len(data_model) * 0.3)
    step_size = round(len(data_model) * 0.1)
    test_window_size = round(len(data_model) * 0.1)
    results = []

    for start_index in range(round(len(data_model) * 0.3), len(data_model) - window_size - test_window_size, step_size):
        train_data = data_model[:round(start_index + window_size)]  #add purged kfold

        test_data_start_index = train_data['index'][-1]
        test_data = data_backtest[data_backtest['index'] > test_data_start_index]
        test_data = test_data[:test_window_size]
        # test_data = data_backtest[test_data_start_index + 1: test_data_start_index + 1 + test_window_size]

        # model.fit(train_data[params['features'] + params['secret_features']], train_data['bin'])

        bt = Backtest(test_data, MyStrategy, cash=100000, commission=.0002, margin=0.02)
        res = bt.run(risk_tolerance=params['risk_tolerance'], model=model, features=params['features'], secret_features=params['secret_features'])
        # bt.plot()


        time.sleep(30)
        segment_results = {
            "timestamp": [t.strftime('%Y-%m-%d %H:%M:%S') for t in test_data.index.tolist()],
            "equity": res['_equity_curve']['Equity'].tolist(),
            "drawdown": res['_equity_curve']['DrawdownPct'].tolist(),
            "statistics": {
                "Return": res['Return [%]'],
                "Volatility": res['Volatility (Ann.) [%]'],
                "Sharpe Ratio": res['Sharpe Ratio'],
                "Max. Drawdown": res['Max. Drawdown [%]']
            }
        }
        results.append(segment_results)

    final = []

    ret = []
    volatility = []
    sharpe_ratio = []
    max_dd = []

    def compute_drawdown(equity_curve):
        peak = equity_curve[0]
        drawdowns = []

        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            drawdowns.append(drawdown)

        return drawdowns

    def combine_equity_curves(results):
        combined_curve = results[0]['equity'].copy()
        all_time = results[0]['timestamp'].copy()

        for segment in results[1:]:
            offset = combined_curve[-1] - segment['equity'][0]

            adjusted_equity = [value + offset for value in segment['equity']]
            combined_curve.extend(adjusted_equity[1:])
            all_time.extend(segment['timestamp'][1:])

        combined_drawdown = compute_drawdown(combined_curve)

        return combined_curve, combined_drawdown, all_time

    combined_curve, combined_drawdown, all_time = combine_equity_curves(results)
    for segment in results:
        ret.append(segment['statistics']['Return'])
        volatility.append(segment['statistics']['Volatility'])
        sharpe_ratio.append(segment['statistics']['Sharpe Ratio'])
        max_dd.append(segment['statistics']['Max. Drawdown'])

    overall_return = (combined_curve[-1] - results[0]['equity'][0]) / results[0]['equity'][0] * 100  # in percentage
    overall_max_dd = max(combined_drawdown)
    volatility = np.mean(volatility)
    sharpe_ratio = np.mean(sharpe_ratio)

    final.append({"timestamp": all_time, "equity": combined_curve, "drawdown": combined_drawdown, 'statistics': {
        "Return": overall_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max. Drawdown": overall_max_dd,
    }})

    return final


def main(best_model, model_data, backtest_data, params):
    df_model = model_data.dropna()
    df_backtest = backtest_data.dropna().copy()
    df_backtest['probs'] = best_model.predict_proba(df_backtest[params['features'] + params['secret_features']])[:, 1]
    signal1 = getSignal(df_backtest, 0.1, df_backtest['probs'], pred=None, numClasses=2, numThreads=None)

    df_backtest.rename(columns={'open': 'Open', 'close': 'Close', 'high': 'High', 'low': 'Low'},
              inplace=True)
    results = walk_forward_backtest(df_model, df_backtest, best_model, params)

    return results


