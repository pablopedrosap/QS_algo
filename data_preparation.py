from concurrent.futures import ProcessPoolExecutor
from random import gauss
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller
import multiprocessing as mp
from tqdm import tqdm


def mpPandasObj(func, pdObj, numThreads, **kwargs):
    parts = [pdObj[1][i::numThreads] for i in range(numThreads)]
    if numThreads == 1:
        out = [func(**{pdObj[0]: part, **kwargs}) for part in parts]
    else:
        with ProcessPoolExecutor(numThreads) as executor:
            out = list(executor.map(lambda part: func(**{pdObj[0]: part, **kwargs}), parts))  # Convert to list

    if isinstance(out[0], pd.DataFrame):
        return pd.concat(out, axis=0)
    elif isinstance(out[0], pd.Series):
        return pd.Series(pd.concat(out, axis=0))
    else:
        raise ValueError('func must return a DataFrame or Series')


def csv(file, tf):
    df = pd.read_csv(file)[50000:60000]
    df.rename(columns={'Timestamp': 'timestamp', 'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp'])
    # df.set_index('timestamp', inplace=True)

    mask = ~((df['timestamp'].dt.dayofweek == 5) | (
            (df['timestamp'].dt.dayofweek == 6) & (df['timestamp'].dt.hour < 17)) | (
                     (df['timestamp'].dt.dayofweek == 4) & (df['timestamp'].dt.hour >= 17)))

    df = df[mask]
    if tf != 'm1':
        df.set_index('timestamp', inplace=True)  #possible error
        df = df.resample('15T').agg({'open': 'first',  #15T
                                     'high': 'max',
                                     'low': 'min',
                                     'close': 'last',
                                     'Volume': 'sum'})
        df.fillna(method='ffill', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def create_dollar_bars(df, m):
    cum_dollar_value = df['volume'].cumsum()
    cum_ticks_over_m = (cum_dollar_value // m)
    df_dollar_grp = df.groupby(cum_ticks_over_m)
    df_dollar = df_dollar_grp.agg(date=('date', 'first'),
                                  open=('open', 'first'),
                                  high=('high', 'max'),
                                  low=('low', 'min'),
                                  close=('close', 'last'),
                                  volume=('volume', 'sum'))
    df_dollar.reset_index(drop=True, inplace=True)
    return df_dollar


# df_dollar = create_dollar_bars(df, 1_000_000)


def getWeights(d, thres):
    w, k = [1.], 1
    while True:
        w_ = -w[-1]/k*(d-k+1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


def fracDiff_FFD(series, d, thres=1e-4):
    w = getWeights(d, thres)
    width = len(w) - 1
    df_ = pd.Series()
    seriesF = series[['close']].fillna(method='ffill').dropna()
    for iloc1 in tqdm(range(width, seriesF.shape[0])):
        loc0, loc1 = seriesF.index[iloc1-width], seriesF.index[iloc1]
        if not np.isfinite(series.loc[loc1, 'close']):
            continue
        df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
    return df_


def calculate_fracdiff(df):
    df = df.sort_values('timestamp', ascending=True)
    original_close = df['close'].copy()
    df['close_log_cum'] = np.log(df['close']).cumsum()
    d = 0.1
    while True:
        fracdiff = fracDiff_FFD(df, d)
        adf_result = adfuller(fracdiff)
        p_value = adf_result[1]
        if p_value < 0.05:
            break
        d += 0.1

    df['close'] = fracdiff
    df['close_original'] = original_close
    return df


def OU(params, data):
    theta, mu, sigma = params
    dt = np.diff(data)
    residuals = dt - theta * (mu - data[:-1])
    neg_log_likelihood = np.sum(residuals ** 2 / sigma ** 2 + np.log(2 * np.pi * sigma ** 2))
    return neg_log_likelihood


def fit_OU_process(prices):
    initial_guess = [0.5, 0.0, 0.5]
    result = minimize(OU, initial_guess, args=(prices,), method='Nelder-Mead')
    theta_hat, mu_hat, sigma_hat = result.x
    return {'forecast': mu_hat, 'hl': theta_hat, 'sigma': sigma_hat}


def mc_simulation(params):
    coeffs, pt, sl, nIter, maxHP, seed = params
    phi = 2 ** (-1. / coeffs['hl'])
    output2 = []
    for iter_ in range(int(nIter)):
        p, hp, count = seed, 0, 0
        while True:
            p = (1 - phi) * coeffs['forecast'] + phi * p + coeffs['sigma'] * gauss(0, 1)
            cP = p - seed
            hp += 1
            if cP > pt or cP < -sl or hp > maxHP:
                output2.append(cP)
                break
    mean, std = np.mean(output2), np.std(output2)
    sharpe_ratio = mean / std
    print(pt, sl)
    return sharpe_ratio


def batch(coeffs, nIter=1e5, maxHP=100, rPT=np.linspace(.5, 10, 20), rSLm=np.linspace(.5, 10, 20), seed=0):
    pool = mp.Pool(mp.cpu_count())
    params_list = [(coeffs, pt, sl, nIter, maxHP, seed) for pt in rPT for sl in rSLm]
    sharpe_ratios = pool.map(mc_simulation, params_list)
    pool.close()
    sharpe_ratios = np.array(sharpe_ratios).reshape(len(rPT), len(rSLm))
    max_sharpe_idx = np.unravel_index(sharpe_ratios.argmax(), sharpe_ratios.shape)
    optimal_pt, optimal_sl = rPT[max_sharpe_idx[0]], rSLm[max_sharpe_idx[1]]
    return optimal_pt, optimal_sl, sharpe_ratios


def main_ptsl(prices):
    rPT = rSLm = np.linspace(0, 10, 21)
    coeffs = fit_OU_process(prices)
    print(coeffs)
    optimal_pt, optimal_sl, sharpe_ratios = batch(coeffs, nIter=1e4, maxHP=100, rPT=rPT, rSLm=rSLm)
    return optimal_pt, optimal_sl


def OTR(df):
    prices = np.array(df['log_returns'])
    return main_ptsl(prices)


'''sampling cusum'''


def apply_cusum_filter(df, threshold_std=1):
    t_events = []
    s_pos = 0
    s_neg = 0
    df['return'].dropna(inplace=True)
    diff = df['return'].diff()
    timestamp = df['timestamp']

    for i in diff.index:
        pos = float(s_pos + diff[i])
        neg = float(s_neg + diff[i])
        s_pos = max(0., pos)
        s_neg = min(0., neg)
        if s_neg < -threshold_std:
            s_neg = 0
            t_events.append(timestamp[i])

        elif s_pos > threshold_std:
            s_pos = 0
            t_events.append(timestamp[i])
    return pd.to_datetime(t_events)


def cusum(df):
    df['return'] = df['close'].pct_change()
    volatility = df['return'].std()
    t_events = apply_cusum_filter(df, threshold_std=volatility)
    return t_events


def getDailyVol(close, span0=100):
    df0 = close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0-1], index=close.index[close.shape[0]-df0.shape[0]:])
    df0 = close.loc[df0.index]/close.loc[df0.values].values-1
    df0 = df0.ewm(span=span0).std()
    return df0


def add_vertical_barrier(t_events, close, num_days=1):
    close_time_values = close.index.values
    t1 = close_time_values.searchsorted((t_events + pd.Timedelta(days=num_days)).values)
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=t_events[:t1.shape[0]])
    return t1


def apply_pt_sl_on_t1(close, events, ptSl, molecule):
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index)
    if ptSl[1] > 0:
        sl = -ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index)
    for loc, t1 in events_['t1'].items():
        df0 = close[loc:t1]
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()
    return out


def getEvents(close, t_events, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    t_events = [event for event in t_events if event in trgt.index]
    trgt = trgt.loc[t_events]
    trgt = trgt[trgt > minRet]

    # get t1 (max holding period)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=t_events)
    if side is None:
        side_, ptSl_ = pd.Series(1., index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        side_, ptSl_ = side.loc[trgt.index], ptSl[:2]

    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset='trgt')
    molecule = events.index
    out = apply_pt_sl_on_t1(close, events, ptSl, molecule)
    out = out.fillna(pd.Timestamp('2099-12-31'))
    t1_new = out.min(axis=1)
    t1_new = t1_new.replace(pd.Timestamp('2099-12-31'), pd.NaT)
    t1_new.dropna(inplace=True)

    # df0 = mpPandasObj(func=apply_pt_sl_on_t1, pdObj=('molecule', events.index), numThreads=numThreads,
    #                                     close=close, events=events, ptSl=[ptSl, ptSl])

    events['t1'] = t1_new
    events.dropna(inplace=True)
    if side is None:
        events = events.drop('side', axis=1)
    return events


def add_barriers(df, t_events):
    df.set_index('timestamp', inplace=True)
    df = df.sort_values(by='timestamp')
    trgt = getDailyVol(df['close'])
    t1 = add_vertical_barrier(t_events, df['close'], num_days=1)
    events = getEvents(df['close'], t_events, ptSl=[3, 1], trgt=trgt, minRet=0, numThreads=10, t1=t1, side=df['side'])

    return df, events


'''weighting samples'''


def getIndMatrix(t1):

    barIx = sorted(set(pd.to_datetime(t1.index)) | set(pd.to_datetime(t1.values)))
    indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
    for i, (t0, t1) in tqdm(enumerate(t1.items())):
        t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
        indM.loc[t0:t1, i] = 1.
    return indM


def getAvgUniqueness(indM):
    c = indM.sum(axis=1)
    u = indM.div(c, axis=0)
    avgU = u[u > 0].mean()
    return avgU


def seqBootstrap(indM, sLength=None):
    if sLength is None:
        sLength = indM.shape[1]
    phi = []
    while len(phi) < sLength:
        avgU = pd.Series()
        for i in tqdm(indM.columns):
            indM_ = indM[phi+[i]]
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
        prob = avgU / avgU.sum()
        phi += [np.random.choice(indM.columns, p=prob)]

    return phi


def mpNumCoEvents(closeidx, t1, molecule):
    t1 = t1.fillna(closeidx[-1])
    t1 = t1[t1 >= molecule[0]]
    t1 = t1.loc[:t1[molecule].max()]
    iloc = closeidx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=closeidx[iloc[0]:iloc[1] + 1])
    for tIn, tOut in t1.items():
        count.loc[tIn:tOut] += 1
    return count.loc[molecule[0]:t1[molecule].max()]


def mpSampleW(t1, numCoEvents, close, molecule):
    ret = np.log(close).diff()
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].items():
        wght.loc[tIn] = (ret.loc[tIn:tOut]/numCoEvents.loc[tIn:tOut]).sum()
    return wght.abs()


def weights(events, df):
    # indM = getIndMatrix(events['t1'])
    # phi = seqBootstrap(indM)
    # avgU = getAvgUniqueness(indM.iloc[:, phi])
    numCoEvents = mpNumCoEvents(df['close'].index, events['t1'], events.index)
    events['w'] = mpSampleW(events['t1'], numCoEvents=numCoEvents, close=df['close'], molecule=events.index)
    # avgU = avgU.reindex(sample_weights.index)
    events['w'] *= events.shape[0] / events['w'].sum()
    return events


def get_bins(events, close):
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    out['w'] = events['w']
    out['trgt'] = events['trgt']
    if 'side' in events_:
        out['ret'] *= events_['side']
        out['bin'] = np.sign(out['ret'])
        out.loc[out['ret'] <= 0, 'bin'] = 0
    return out


def drop_labels(events, minPtc=.05):
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > minPtc or df0.shape[0] < 3:
            break
        events = events[events['bin'] != df0.idxmin()]
    return events


def secondary_features(df, params):
    def secret_features(df):
        params['secret_features'] = []
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_ret'].rolling(window=21).std() * np.sqrt(21)
        df['autocorr_1'] = df['log_ret'].rolling(window=21).apply(lambda x: x.autocorr(lag=1), raw=False)
        params['secret_features'] += ['volatility', 'autocorr_1']
        return df

    df = secret_features(df)
    return df
