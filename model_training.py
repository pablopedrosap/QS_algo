import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection._split import _BaseKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
import pickle


class MyPipeline(Pipeline):
    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0]+'__sample_weight'] = sample_weight
        return super(MyPipeline, self).fit(X, y, **fit_params)


class PurgedKFold(_BaseKFold):
    def __init__(self, n_splits=3, t1=None, pctEmbargo=0):
        if not isinstance(t1, pd.Series):
            raise ValueError('label through dates must be a pandas series')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        if (X.index == self.t1.index).sum() != len(self.t1):
            pass
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0]*self.pctEmbargo)
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for i, j in test_starts:
            t0 = self.t1.index[i]
            test_indices = indices[i:j]
            maxT1Idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            train_indices = np.concatenate((train_indices, indices[maxT1Idx+mbrg:]))
            yield train_indices, test_indices


'''OPTUNA FOR DEEP LEARNING'''


def clfHyperFit(feat, lbl, t1, pipe_clf, param_grid, cv=3, bagging=[0, None, 1.], rndSearchIter=0, n_jobs=-1, pctEmbargo=0,
                **fit_params):
    if set(lbl.values) == {0, 1}:
        scoring = 'f1'
    else:
        scoring = 'neg_log_loss'
    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)

    if rndSearchIter == 0:
        gs = GridSearchCV(estimator=pipe_clf, param_grid=param_grid, scoring=scoring, cv=inner_cv,
                          n_jobs=n_jobs)
    else:
        gs = RandomizedSearchCV(estimator=pipe_clf, param_distributions=param_grid, scoring=scoring, cv=inner_cv,
                                n_jobs=n_jobs, n_iter=rndSearchIter)
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_
    if bagging[1] > 0:
        gs = BaggingClassifier(estimator=MyPipeline(gs.steps), n_estimators=int(bagging[0]),
                               max_samples=float(bagging[1]), max_features=float(bagging[2]), n_jobs=n_jobs)
        gs = gs.fit(feat, lbl)
        # gs = gs.fit(feat, lbl, sample_weight=fit_params[gs.base_estimator.steps[-1][0] + '__sample_weight'])
        gs = Pipeline([('bag', gs)])
    return gs


def cvScore(clf, X, y, cv=None, cvGen=None, sample_weight=None, t1=None, pctEmbargo=None, scoring='neg_log_loss'):
    if cvGen is None:
        cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)
    score = []
    score_ = []
    for train, test in tqdm(cvGen.split(X=X)):
        fit = clf.fit(X=X.iloc[train, :], y=y.iloc[train], sample_weight=sample_weight.iloc[train].values)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X.iloc[test, :])
            score_ = -log_loss(y.iloc[test], prob, sample_weight=sample_weight.iloc[test].values, labels=clf.classes_)
        score.append(score_)
    return np.array(score)


def getRealData(real_data, features):
    real_data = real_data.dropna().copy()
    trnsX = real_data[features]
    cont = real_data[['bin']].copy()
    # cont['w'] = 1. / cont.shape[0]
    cont['t1'] = pd.Series(cont.index, index=cont.index) #this is wrong
    return trnsX, cont


def testFunc(df, params, n_estimators=100, cv=10, tuning=True):
    trnsX, cont = getRealData(df, params['features'] + params['secret_features'])

    if params['ml_model'] == 'random_forest':
        pipe_clf = Pipeline([
            ('scl', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100))
        ])
    param_grid = {'clf__n_estimators': [10, 100, 500], 'clf__max_depth': [None, 20, 30, 50]}
    best_model = clfHyperFit(trnsX, cont['bin'], cont['t1'], pipe_clf, param_grid, rndSearchIter=10, cv=cv,
                             bagging=[n_estimators, 1., 1.], n_jobs=-1, pctEmbargo=0)


    # method = 'MDI'
    # imp = featImpMDI(best_model, featNames=trnsX.columns) if method == 'MDI' else None
    # oos = cvScore(best_model, X=trnsX, y=cont['successful'], cv=cv, t1=cont['t1'],
    #                 pctEmbargo=0, scoring='neg_log_loss').mean()
    # print(oos)
    # predictions = best_model.predict(trnsX)
    # actuals = cont['bin']
    # sharpe_ratio = sharpeRatio(predictions, actuals)
    return best_model
    
