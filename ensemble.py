import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-n', default=2000, type=int, help='Number of boosting rounds.')
parser.add_argument('-s', action='store_true', default=False,
                    help='Set this flag for submission. Default is cross validation.')
args = parser.parse_args()
num_rounds = args.n

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
    'verbose': 0
}

xgb_param = {
    'objective': 'multi:softprob',
    'eta': 0.02,
    'max_depth': 4,
    'silent': 1,
    'num_class': 3,
    'eval_metric': "mlogloss",
    'min_child_weight': 1,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'seed': 2017,
    'nthread': 8
}


class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models
        self.n_class = 3    # Number of classes/labels

    def __str__(self):
        return 'Stacker: {} \nBase models: {}'.format(self.stacker, self.base_models)

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=2018)
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], skf.get_n_splits(X)))
            for j, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                S_train[test_idx, i] = clf.predict(X_holdout)[:]
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(1)
        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred

    def fit_predict_proba(self, X, y, T, y_valid=None):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=2018)
        S_train = np.zeros((X.shape[0], len(self.base_models), self.n_class))
        S_test = np.zeros((T.shape[0], len(self.base_models), self.n_class))
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], skf.get_n_splits(X), self.n_class))
            for j, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                if isinstance(clf, LGBMClassifier):
                    lgb_train = lgb.Dataset(X_train, y_train)
                    gbm = lgb.train(lgb_param, lgb_train, num_boost_round=num_rounds)
                    S_train[test_idx, i] = gbm.predict(X_holdout, num_iteration=gbm.best_iteration)
                    S_test_i[:, j] = gbm.predict(T, num_iteration=gbm.best_iteration)
                elif isinstance(clf, XGBClassifier):
                    xgb_train = xgb.DMatrix(X_train, label=y_train)
                    gbm = xgb.train(xgb_param, xgb_train, num_rounds)
                    S_train[test_idx, i] = gbm.predict(xgb.DMatrix(X_holdout))
                    S_test_i[:, j] = gbm.predict(xgb.DMatrix(T))
                else:
                    clf.fit(X_train, y_train)
                    S_train[test_idx, i] = clf.predict_proba(X_holdout)[:]
                    S_test_i[:, j] = clf.predict_proba(T)[:]
            S_test[:, i] = S_test_i.mean(1)
        S_train = S_train.reshape((S_train.shape[0], S_train.shape[1] * S_train.shape[2]))
        S_test = S_test.reshape((S_test.shape[0], S_test.shape[1] * S_test.shape[2]))
        if y_valid is None:
            # No validation label is passed in, this is for submission
            np.savetxt('data/s_train.csv', np.c_[y, S_train], delimiter=',')
            np.savetxt('data/s_test.csv', S_test, delimiter=',')
        else:
            # Self validation, save 2nd layer features
            np.savetxt('data/s_train_part.csv', np.c_[y, S_train], delimiter=',')
            np.savetxt('data/s_valid.csv', np.c_[y_valid, S_test], delimiter=',')
        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict_proba(S_test)[:]
        return y_pred


X_train = pd.read_csv("data/train_python.csv", encoding='utf-8')
y_train = X_train['interest_level']
X_train = X_train.drop('interest_level', axis=1)

xgb_model = XGBClassifier()
lgb_model = LGBMClassifier()
rf_model = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=2018)
gbt_model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, random_state=2018)
ensemble = Ensemble(3, gbt_model, [xgb_model, lgb_model, rf_model])

if args.s:
    X_test = pd.read_csv("data/test_python.csv")
    y_pred = ensemble.fit_predict_proba(X_train, y_train, X_test)
    np.savetxt('submission/submission.csv', np.c_[X_test['listing_id'], y_pred[:, [2, 1, 0]]], delimiter=',',
               header='listing_id,high,medium,low', fmt='%d,%.16f,%.16f,%.16f', comments='')
else:
    X_train_part, X_valid, y_train_part, y_valid = \
        train_test_split(X_train, y_train, test_size=0.3, stratify=y_train, random_state=2018)
    y_pred = ensemble.fit_predict_proba(X_train_part, y_train_part, X_valid, y_valid)
    print(log_loss(y_valid, y_pred))
