import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


def discreteSignal(signal0, stepSize):
    signal1 = (signal0 / stepSize).round() * stepSize
    signal1[signal1 > 1] = 1
    signal1[signal1 < -1] = -1
    return signal1


def mpAvgActiveSignals(signals, molecule):
    out = pd.Series()
    for loc in molecule:
        df0 = (signals.index.values <= loc) & ((loc < signals['t1']) | pd.isnull(signals['t1']))
        act = signals[df0].index
        if len(act) > 0:
            out[loc] = signals.loc[act, 'signal'].mean()
        else:
            out[loc] = 0
    return out


def avgActiveSignal(signals, numThreads):
    tPnts = set(signals['t1'].dropna().values)
    tPnts = tPnts.union(signals.index.values)
    tPnts = list(tPnts)
    tPnts.sort()
    out = mpAvgActiveSignals(signals=signals, molecule=tPnts)
    return out


def getSignal(events, stepSize, prob, pred, numClasses, numThreads, **kargs):
    pred = 1
    if prob.shape[0] == 0:
        return pd.Series()
    signal0 = (prob - 1. / numClasses) / (prob * (1. - prob)) ** .5
    signal0 = pd.Series(pred * (2 * norm.cdf(signal0) - 1), index=prob.index)
    signal0.index = prob.index
    if 'side' in events:
        signal0 *= events.loc[signal0.index, 'side']
    df0 = signal0.to_frame('signal').join(events[['t1']], how='left')
    df0 = avgActiveSignal(df0, numThreads)

    print(signal0[1100:1150])
    print(df0[1100:1150])
    signal1 = discreteSignal(signal0=df0, stepSize=stepSize)
    print(signal1[1100:1150])
    print(signal1.value_counts())
    return signal1







def betSize(w, x):
    return x * (w+x**2)**-.5


def getTPos(w, f, mP, maxPos):
    return int(betSize(w, f-mP)*maxPos)


def invPrice(f, w, m):
    return f-m*(w/(1-m**2))**.5


def limitPrice(tPos, pos, f, w, maxPos):
    sgn = 1 if tPos >= pos else -1
    lP = 0
    for j in range(abs(pos+sgn), abs(tPos+1)):
        lP += invPrice(f, w, j/float(maxPos))
    lP /= tPos - pos
    return lP


def getW(x, m):
    return x**2*(m**-2-1)


def position_size():
    pos, maxPos, mP, f, wParams = 0, 100, 100, 175, {'divergence': 10, 'm': .95}
    w = getW(wParams['divergence'], wParams['m'])
    tPos = getTPos(w, f, mP, maxPos)
    print(tPos)
    lP = limitPrice(tPos, pos, f, w, maxPos)
    return tPos




