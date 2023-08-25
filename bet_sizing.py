import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


def betSize(p, N):
    print(p)
    if isinstance(p, (list, np.ndarray)):
        p = pd.Series(p[:, 1])
    z = (p - 1 / N) / np.sqrt(p * (1 - p))
    bet = 2 * norm.cdf(z) - 1
    return bet


def calculate_bet_sizes(probs, t1):
    bet_sizes = betSize(probs, N=2)
    print(bet_sizes)

    df = pd.DataFrame({
        'signal': bet_sizes,
        't1': t1
    })

    df_long = df[df['signal'] > 0]
    df_short = df[df['signal'] < 0]

    if df_long.empty:
        date_range_long = None
    else:
        active_periods_long = df_long['t1'].reset_index()
        active_periods_long.columns = ['start', 'end']
        date_range_long = pd.date_range(start=active_periods_long['start'].min(), end=active_periods_long['end'].max())
    if df_short.empty:
        date_range_short = None
    else:
        active_periods_short = df_short['t1'].reset_index()
        active_periods_short.columns = ['start', 'end']
        date_range_short = pd.date_range(start=active_periods_short['start'].min(),
                                         end=active_periods_short['end'].max())

    long_bets = pd.DataFrame(index=date_range_long)
    long_bets['num_long_bets'] = long_bets.index.to_series().apply(
        lambda date: ((active_periods_long['start'] <= date) & (date <= active_periods_long['end'])).sum())
    short_bets = pd.DataFrame(index=date_range_short)
    short_bets['num_short_bets'] = short_bets.index.to_series().apply(
        lambda date: ((active_periods_short['start'] <= date) & (date <= active_periods_short['end'])).sum())
    long_bets = long_bets.reindex(short_bets.index, fill_value=0)
    new_series = pd.Series(long_bets['num_long_bets'].values - short_bets['num_short_bets'].values)
    data = new_series.values.reshape(-1, 1)

    if len(data) > 1:
        gmm = GaussianMixture(n_components=2)
        gmm.fit(data)

        def gmm_cdf(x, gmm):
            cdf = 0
            for i in range(gmm.n_components):
                weight = gmm.weights_[i]
                mean = gmm.means_[i, 0]
                cov = gmm.covariances_[i, 0]
                cdf += weight * norm(loc=mean, scale=np.sqrt(cov)).cdf(x)
            return cdf

        F_0 = gmm_cdf(0, gmm)
        bet_sizes = pd.Series(index=new_series.index)
        for i, x in new_series.items():
            F_x = gmm_cdf(x, gmm)
            if x >= 0:
                bet_sizes[i] = (F_x - F_0) / (1 - F_0)
            else:
                bet_sizes[i] = (F_x - F_0) / F_0
    else:
        # Skip this iteration and proceed to the next one, or
        # return a default bet size
        bet_sizes = pd.Series([0.5], index=new_series.index)
    return bet_sizes

