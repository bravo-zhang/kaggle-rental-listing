import numpy as np
import pandas as pd
import lightgbm as lgb
import argparse
from sklearn.externals import joblib

parser = argparse.ArgumentParser(description='Light Gradient Boosting Tree Model.')
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

X_train = pd.read_csv("data/train_python.csv", encoding='utf-8')
y_train = X_train['interest_level']
X_train = X_train.drop('interest_level', axis=1)
lgb_train = lgb.Dataset(X_train, y_train)

if args.s:
    clf = lgb.train(lgb_param, lgb_train, num_boost_round=num_rounds)
    joblib.dump(clf, 'checkpoint/lgb.pkl')
    X_test = pd.read_csv("data/test_python.csv", encoding='utf-8')
    pred = clf.predict(X_test)
    np.savetxt('submission/submission.csv', np.c_[X_test['listing_id'], pred[:, [2, 1, 0]]], delimiter=',',
               header='listing_id,high,medium,low', fmt='%d,%.16f,%.16f,%.16f', comments='')
else:
    cv_results = lgb.cv(lgb_param, lgb_train, num_boost_round=num_rounds, nfold=5, stratified=True,
                        early_stopping_rounds=20, callbacks=[lgb.callback.print_evaluation(show_stdv=True)])
