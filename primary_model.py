import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks, savgol_filter


def draw_line_chart(df, support_resistance=None):
    plt.plot(df.index, df['close'], label='Close Price')

    if support_resistance is not None:
        for level in support_resistance:
            plt.axhline(y=level, color='r', linestyle='--')

    plt.xlabel('Timestamp')
    plt.ylabel('Close Price')
    plt.title('Close Price with Support and Resistance Levels')
    plt.legend()
    plt.show()


# def draw_candle_chart(df, support_resistance=None):
#     fig, ax = mpf.plot(df, type='candle', returnfig=True)
#
#     if support_resistance is not None:
#         for level in support_resistance:
#             ax[0].axhline(y=level, color='r', linestyle='--')
#
#     plt.show()


def primary_features(df, params, live=False):
    def technical_features(df, params):
        if 'moving_average' in params['technical_feature']:
            df['ma_10'] = df['close'].rolling(window=10).mean()
            df['ma_50'] = df['close'].rolling(window=50).mean()
            df['signalMA'] = 0
            df.loc[df['ma_10'] > df['ma_50'], 'signalMA'] = 1
            df.loc[df['ma_10'] < df['ma_50'], 'signalMA'] = -1
            params['features'] += ['signalMA']
        if 'bollinger_bands' in params['technical_feature']:
            def bollinger_bands(df, n, m):
                data = (df['high'] + df['low'] + df['close']) / 3
                B_MA = pd.Series((data.rolling(n, min_periods=n).mean()), name='B_MA')
                sigma = data.rolling(n, min_periods=n).std()
                BU = pd.Series((B_MA + m * sigma), name='BU')
                BL = pd.Series((B_MA - m * sigma), name='BL')
                df = df.join(B_MA)
                df = df.join(BU)
                df = df.join(BL)
                return df

            df = bollinger_bands(df, 20, 1)
            df['signalBB'] = 0
            df.loc[df['close'] > df['BU'], 'signalBB'] = -1
            df.loc[df['close'] < df['BL'], 'signalBB'] = 1
            params['features'] += ['signalBB']

        if 'support_resistance' in params['technical_feature']:
            df.reset_index(drop=True, inplace=True)
            df['close_smooth'] = savgol_filter(df.close, 500, 5)

            maxima_indices = argrelextrema(df.close.values, np.greater)[0]
            minima_indices = argrelextrema(df.close.values, np.less)[0]

            maxima_values = df.close[maxima_indices]
            minima_values = df.close[minima_indices]

            extrema_prices = np.concatenate((maxima_values, minima_values), axis=0)

            peaks_range = [2, 4]
            num_peaks = -999
            interval = extrema_prices[0] / 10000
            bandwidth = interval

            while num_peaks < peaks_range[0] or num_peaks > peaks_range[1]:
                initial_price = extrema_prices[0]
                kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(
                    extrema_prices.reshape(-1, 1))
                price_range = np.linspace(min(extrema_prices), max(extrema_prices), 1000).reshape(-1, 1)
                pdf = np.exp(kde.score_samples(price_range))
                peaks = find_peaks(pdf)[0]

                num_peaks = len(peaks)
                bandwidth += interval
                if bandwidth > 100 * interval:
                    break
            # draw_line_chart(df, price_range[peaks])
            df = df.copy()
            df['signalSupRes'] = 0
            threshold = 0.02  # For example, 2% of the price for how close
            for i in range(len(df)):
                close_price = df.close.iloc[i]
                for level in price_range[peaks]:
                    if level * (1 - threshold) <= close_price <= level * (1 + threshold):
                        if close_price < level:
                            df.loc[i, 'signalSupRes'] = 1
                        else:
                            df.loc[i, 'signalSupRes'] = -1
            params['features'] += ['signalSupRes']

        if 'higher_highs_lows' in params['technical_feature']:
            window_size = int(len(df) * 0.02)
            if window_size % 2 == 0:
                window_size += 1
            distance = int(window_size / 2)

            df['close_smooth'] = savgol_filter(df.close, window_size, 5)
            peaks_idx, _ = find_peaks(df.close_smooth.values, distance=distance, width=3)
            troughs_idx, _ = find_peaks(-1 * df.close_smooth.values, distance=50, width=3)

            # Initialize signals
            df['signalZigZag'] = 0
            current_signal = 0
            df = df.copy()

            for i in range(1, len(df)):
                previous_peaks = peaks_idx[peaks_idx < i]
                previous_troughs = troughs_idx[troughs_idx < i]

                if len(previous_peaks) > 0 and i in peaks_idx and df.close_smooth.iloc[i] > df.close_smooth.iloc[
                    previous_peaks[-1]]:
                    current_signal = 1

                if len(previous_troughs) > 0 and current_signal == 1 and df.close_smooth.iloc[i] < df.close_smooth.iloc[
                    previous_troughs[-1]]:
                    current_signal = 0

                if len(previous_troughs) > 0 and i in troughs_idx and df.close_smooth.iloc[i] < df.close_smooth.iloc[
                    previous_troughs[-1]]:
                    current_signal = -1

                if len(previous_peaks) > 0 and current_signal == -1 and df.close_smooth.iloc[i] > df.close_smooth.iloc[
                    previous_peaks[-1]]:
                    current_signal = 0

                df.loc[i, 'signalZigZag'] = current_signal
            params['features'] += ['signalZigZag']

            plt.plot(df.index, df['close_smooth'].values, label='Smoothed Close Prices', color='black')

            # Plotting the peaks and troughs
            peaks_idx_to_plot = peaks_idx  # Selecting every 10th peak, adjust as needed
            troughs_idx_to_plot = troughs_idx  # Selecting every 10th trough, adjust as needed
            plt.scatter(peaks_idx_to_plot, df['close_smooth'][peaks_idx_to_plot], color='r', label='Peaks', s=90)
            plt.scatter(troughs_idx_to_plot, df['close_smooth'][troughs_idx_to_plot], color='g', label='Troughs', s=90)

            # Adding buy and sell signals
            buy_signals = df[df['signalZigZag'] == 1].index[::10]  # Selecting every 10th buy signal, adjust as needed
            sell_signals = df[df['signalZigZag'] == -1].index[::10]  # Selecting every 10th sell signal, adjust as needed
            plt.scatter(buy_signals, df['close_smooth'][buy_signals], color='b', label='Buy', s=80, marker='^')
            plt.scatter(sell_signals, df['close_smooth'][sell_signals], color='y', label='Sell', s=80, marker='v')

            plt.xlabel('Index')
            plt.ylabel('Close Price')
            plt.title('Smoothed Close Prices with Peaks and Troughs')
            plt.legend()
            plt.show()

        return df

    def fundamental_features(df, fundamental):
        return df
        pass

    def sentiment_features(df, params):
        if 'retail_sentiment' in params['sentiment_feature']:
            df.set_index('timestamp')
            perc = pd.read_csv(f'/Users/pablopedrosa/PycharmProjects/QS_algorithm/downloads/{params["asset_name"][0].lower()}_sent.csv')
            perc['time'] = (pd.to_datetime(perc['time']) - pd.Timedelta(days=1)).dt.date
            df['date'] = df['timestamp'].dt.date
            df = pd.merge(df, perc, left_on='date', right_on='time', how='left')
            df['long_percentage'].fillna(method='ffill', inplace=True)
            df.drop(columns=['time', 'date'], inplace=True)
            df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
            df['signalContrarianSent'] = 0
            df.loc[df['long_percentage'] < 0.5, 'signalContrarianSent'] = 1
            df.loc[df['long_percentage'] > 0.5, 'signalContrarianSent'] = -1
            params['features'] += ['signalContrarianSent']
        return df

    if live:
        params['features'] = []
        df = technical_features(df, params)

    else:
        df.dropna(inplace=True)
        df.index = df['timestamp']
        params['features'] = []

        df.reset_index(drop=True, inplace=True)
        df = technical_features(df, params)
        df = fundamental_features(df, params)
        df = sentiment_features(df, params)

        def get_side(row, features):
            signals = [row[feature] for feature in features]
            side = 1 if sum(signals) == len(features) else -1 if sum(signals) == -len(features) else 0
            return side

        df['side'] = df.apply(lambda row: get_side(row, params['features']), axis=1)
        df = df.reset_index()
        # print(f"long signals: {len(df[df['side'] == 1]) / len(df[df['side'] == -1])} %")
        # print(len(df[df['side'] == -1]))
        df.index = df['timestamp']
    return df
