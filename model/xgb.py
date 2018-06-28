import numpy as np
import pandas as pd
import xgboost as xgb
import argparse
from sklearn.externals import joblib

parser = argparse.ArgumentParser(description='Extreme Gradient Boosting Tree Model.')
parser.add_argument('-n', default=2000, type=int, help='Number of boosting rounds.')
parser.add_argument('-s', action='store_true', default=False,
                    help='Set this flag for submission. Default is cross validation.')
args = parser.parse_args()

num_rounds = args.n
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

X_train = pd.read_csv("data/train_python.csv", encoding='utf-8')
y_train = X_train['interest_level']
X_train = X_train.drop('interest_level', axis=1)
xgb_train = xgb.DMatrix(X_train, label=y_train)

if args.s:
    clf = xgb.train(xgb_param, xgb_train, num_boost_round=num_rounds)
    joblib.dump(clf, 'checkpoint/xgb.pkl')
    X_test = pd.read_csv("data/test_python.csv", encoding='utf-8')
    pred = clf.predict(xgb.DMatrix(X_test))
    np.savetxt('submission/submission.csv', np.c_[X_test['listing_id'], pred[:, [2, 1, 0]]], delimiter=',',
               header='listing_id,high,medium,low', fmt='%d,%.16f,%.16f,%.16f', comments='')
else:
    xgb.cv(xgb_param, xgb_train, num_boost_round=num_rounds, nfold=5, stratified=True,
           callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
