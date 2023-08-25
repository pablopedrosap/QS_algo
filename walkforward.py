import os
import sys
import pandas as pd
import numpy as np
from numpy import array
from scipy.stats import norm
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm
from bet_sizing import calculate_bet_sizes
import matplotlib.pyplot as plt
import re
import json
import subprocess
from backtesting import Backtest, Strategy


# def walk_forward_backtest(model, data, params, initial_investment, pnl):
#     investment = initial_investment + pnl
#     results = []
#     positions = []
#     decisions = []
#
#     dates = data.index.unique()
#     data['trgt_pips'] = data['trgt'] * 10000
#
#     for i in range(len(dates)):
#         current_date = dates[i]
#         current_data = data.loc[data.index == current_date]
#         SL_pips = current_data['trgt_pips'].values[0]
#         risk = {'high': 0.02, 'low': 0.001}.get(params['risk_tolerance'], 0.005)
#
#         trade_direction = get_trade_direction(current_data, params['features'])
#         if trade_direction != 0 and investment > 0:
#             # print('trade')
#             # prob_success = model.predict_proba(current_data[params['features'] + params['secret_features']])[0][1]
#             # print(prob_success)
#             # bet_size = prob_success if prob_success > 0.6 else 0
#             bet_size = 1
#             number_of_lots = (investment * risk) / (SL_pips * 10) * bet_size
#             positions.append((number_of_lots, current_data['close_original'].item(),
#                               'long' if trade_direction == 1 else 'short'))
#             decisions.append(trade_direction)
#
#         positions_copy = positions.copy()
#         for position in positions_copy:
#             number_of_lots, entry_price, position_type = position
#             tp_level = entry_price + (3 * SL_pips / 10000 if position_type == 'long' else -3 * SL_pips / 10000)
#             sl_level = entry_price - (SL_pips / 10000 if position_type == 'long' else -SL_pips / 10000)
#
#             current_price = current_data['close_original'].item()
#             if (position_type == 'long' and (current_price >= tp_level or current_price <= sl_level)) or \
#                     (position_type == 'short' and (current_price <= tp_level or current_price >= sl_level)):
#                 pnl_value = number_of_lots * 100000 * ((current_price - entry_price) if position_type == 'long'
#                                                      else (entry_price - current_price))
#                 investment += pnl_value
#                 positions.remove(position)
#
#         results.append(investment - initial_investment)
#
#     results = pd.Series(results, index=dates)
#     return results
#
#
# def backtest_statistics(ret, df, params):
#     def getHHI(betRet):
#         if betRet.shape[0] <= 2:
#             return np.nan
#         wght = betRet / betRet.sum()
#         hhi = (wght ** 2).sum()
#         hhi = (hhi - betRet.shape[0] ** -1) / (1. - betRet.shape[0] ** -1)
#         return hhi
#
#     def computeDD_TuW(series, dollars=False):
#         df0 = series.to_frame('pnl')
#         df0['hwm'] = series.expanding().max()
#         df1 = df0.groupby('hwm').min().reset_index()
#         df1.columns = ['hwm', 'min']
#         df1.index = df0['hwm'].drop_duplicates(keep='first').index
#         df1 = df1[df1['hwm'] > df1['min']]
#         if dollars:
#             dd = df1['hwm'] - df1['min']
#         else:
#             dd = 1 - df1['min'] / df1['hwm']
#         tuw = ((df1.index[1:] - df1.index[:-1]) / np.timedelta64(1, 'Y')).values
#         tuw = pd.Series(tuw, index=df1.index[:-1])
#         return dd, tuw
#
#     def binHR(sl, pt, freq, tSR):
#         a = (freq + tSR**2) * (pt - sl)**2
#         b = (2*freq*sl - tSR**2*(pt - sl))*(pt - sl)
#         c = freq*sl**2
#         p = (-b+(b**2-4*a*c)**.5)/(2.*a)
#         return p
#
#     def probFailure(ret, freq, tSR):
#         rPos, rNeg = ret[ret > 0].mean(), ret[ret <= 0].mean()
#         p = ret[ret > 0].shape[0] / float(ret.shape[0])
#         thresP = binHR(rNeg, rPos, freq, tSR)
#         risk = norm.cdf(thresP, p, p*(1-p))
#         return risk
#
#     tSR, freq = 2., 260
#     probF = probFailure(ret, freq, tSR)
#     tHHI = getHHI(ret.groupby(pd.Grouper(freq='M')).count())
#     dd, tuw = computeDD_TuW(ret)
#     ret_decimal = ret / 100
#     annualized_avg_return = (1 + ret_decimal.mean())**252 - 1
#     annualized_SR = ret_decimal.mean() / ret_decimal.std() * np.sqrt(252)
#     return {
#         'ret': ret.values.tolist(),
#         'HHI': float(tHHI),
#         'Drawdown': dd.tolist(),
#         'Time Under Water': tuw.tolist(),
#         'Annualized Sharpe Ratio': float(annualized_SR),
#         'Annualized Average Return': float(annualized_avg_return),
#         'Probability of failure': float(probF)
#     }


# def main(best_model, df, params):
#     params['initial_investment']: 100000
#     df = df.dropna()
#     window_size = round(len(df) * 0.3)
#     step_size = round(len(df) * 0.1)
#     test_window_size = round(len(df) * 0.1)
#     investment = 100000
#     results_values = pd.Series(dtype=float)
#
#     for start_index in tqdm(range(round(len(df) * 0.3), len(df) - window_size - test_window_size, step_size)):
#         train_data = df[:round((start_index + window_size) * 0.95)]  # include purgedkfold
#         test_data = df[start_index + window_size:start_index + window_size + test_window_size]
#
#         model = best_model.fit(train_data[params['features'] + params['secret_features']], train_data['bin'])
#         pnl = walk_forward_backtest(model, test_data, params, initial_investment=investment,
#                                     pnl=results_values.iloc[-1] if len(results_values) > 0 else 0)
#         results_values = pd.concat([results_values, pnl])
#
#     results_values = results_values / params['initial_investment']
#     plt.plot(results_values.index, results_values.values)
#     plt.show()
#     stats = backtest_statistics(results_values, df, params)
#     return stats


def get_trade_direction(data, features):
    unique_values = set(data[feature] for feature in features)
    if len(unique_values) == 1 and 0 not in unique_values:
        return unique_values.pop()
    return 1    # THIS IS WRONG


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
            'trgt_pips': 0.02,
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
            position_value = 100000  # Just a placeholder

            # Implement METAMODEL BET SIZING
            if trade_direction == 1:
                self.buy(size=position_value, tp=current_data.Close + SL_pips*3, sl=current_data.Close - SL_pips)
            else:
                self.sell(size=position_value, tp=current_data.Close - SL_pips*3, sl=current_data.Close + SL_pips)


def walk_forward_backtest(data_model, data_backtest, model, params):
    window_size = round(len(data_model) * 0.3)
    step_size = round(len(data_model) * 0.1)
    test_window_size = round(len(data_model) * 0.1)
    results = []

    for start_index in range(round(len(data_model) * 0.3), len(data_model) - window_size - test_window_size, step_size):
        train_data = data_model[:round(start_index + window_size)]  #add purged kfold
        test_data_start_index = data_backtest.index.get_loc(train_data.index[-1])
        test_data = data_backtest[test_data_start_index + 1: test_data_start_index + 1 + test_window_size]

        model.fit(train_data[params['features'] + params['secret_features']], train_data['bin'])
        bt = Backtest(test_data, MyStrategy, cash=100000, commission=.002)
        res = bt.run(risk_tolerance=params['risk_tolerance'], model=model, features=params['features'], secret_features=params['secret_features'])

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
        "Max. Drawdown": overall_max_dd
    }})

    return final


def main(best_model, model_data, backtest_data, params):
    df_model = model_data.dropna()
    df_backtest = backtest_data.dropna().copy()

    df_backtest.rename(columns={'open': 'Open', 'close': 'Close', 'high': 'High', 'low': 'Low'},
              inplace=True)
    results = walk_forward_backtest(df_model, df_backtest, best_model, params)
    return results


