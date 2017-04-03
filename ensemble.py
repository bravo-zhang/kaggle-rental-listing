import numpy as np
import lightgbm as lgb
import xgboost as xgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold

num_rounds = 2000
lgb_param = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_classes': 3,
    'metric': {'multi_logloss'},
    'num_leaves': 55,
    'learning_rate': 0.005,
    'feature_fraction': 0.82,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'nthread': 8
}
xgb_param = {}
xgb_param['objective'] = 'multi:softprob'
xgb_param['eta'] = 0.02
xgb_param['max_depth'] = 4
xgb_param['silent'] = 1
xgb_param['num_class'] = 3
xgb_param['eval_metric'] = "mlogloss"
xgb_param['min_child_weight'] = 1
xgb_param['subsample'] = 0.7
xgb_param['colsample_bytree'] = 0.7
xgb_param['seed'] = 2017
xgb_param['nthread'] = 8


class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models
        self.n_class = 3    # Number of classes/labels

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=2017)
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], kf.get_n_splits(X)))
            for j, (train_idx, test_idx) in enumerate(kf.split(X)):
                print len(train_idx), len(test_idx)
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(1)
        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred

    def fit_predict_proba(self, X, y, T, y_valid=None):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=2017)
        S_train = np.zeros((X.shape[0], len(self.base_models), self.n_class))
        S_test = np.zeros((T.shape[0], len(self.base_models), self.n_class))
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], kf.get_n_splits(X), self.n_class))
            for j, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                if isinstance(clf, LGBMClassifier):
                    lgb_train = lgb.Dataset(X_train, y_train)
                    gbm = lgb.train(lgb_param, lgb_train, num_boost_round=num_rounds)
                    y_pred = gbm.predict(X_holdout, num_iteration=gbm.best_iteration)
                    S_train[test_idx, i] = y_pred
                    S_test_i[:, j] = gbm.predict(T, num_iteration=gbm.best_iteration)
                elif isinstance(clf, XGBClassifier) :
                    xgtrain = xgb.DMatrix(X_train, label=y_train)
                    gbm = xgb.train(xgb_param, xgtrain, num_rounds)
                    y_pred = gbm.predict(xgb.DMatrix(X_holdout))
                    S_train[test_idx, i] = y_pred
                    S_test_i[:, j] = gbm.predict(xgb.DMatrix(T))
                else:
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict_proba(X_holdout)[:]
                    S_train[test_idx, i] = y_pred
                    S_test_i[:, j] = clf.predict_proba(T)[:]
            S_test[:, i] = S_test_i.mean(1)
        S_train = S_train.reshape((S_train.shape[0], S_train.shape[1] * S_train.shape[2]))
        S_test = S_test.reshape((S_test.shape[0], S_test.shape[1] * S_test.shape[2]))
        if y_valid is None:
            # No validation label is passed in, this is for submission
            np.savetxt('s_train.csv', np.c_[y, S_train], delimiter=',')
            np.savetxt('s_test.csv', S_test, delimiter=',')
        else:
            # Self validation, save 2nd layer features
            np.savetxt('s_train_part.csv', np.c_[y, S_train], delimiter=',')
            np.savetxt('s_valid.csv', np.c_[y_valid, S_test], delimiter=',')
        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict_proba(S_test)[:]
        return y_pred
