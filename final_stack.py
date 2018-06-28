import numpy as np
import pandas as pd
import argparse
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

parser = argparse.ArgumentParser(description='Logistic Regression Model.')
parser.add_argument('-s', action='store_true', default=False,
                    help='Set this flag for submission. Default is cross validation.')
args = parser.parse_args()

X_train = pd.read_csv("data/s_train.csv", header=None, encoding='utf-8')
y_train = X_train.iloc[:,0]
X_train = X_train.iloc[:,1:]

clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, random_state=2018)
if args.s:
    clf.fit(X_train, y_train)
    X_test = pd.read_csv("data/s_test.csv", header=None, encoding='utf-8')
    pred = clf.predict_proba(X_test)
    listing_id = pd.read_csv("data/test_python.csv", usecols=['listing_id'], encoding='utf-8')
    np.savetxt('submission/submission.csv', np.c_[listing_id, pred[:, [2, 1, 0]]], delimiter=',',
               header='listing_id,high,medium,low', fmt='%d,%.16f,%.16f,%.16f', comments='')
else:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
    scores = cross_val_score(clf, X_train, y_train, scoring='neg_log_loss', cv=cv, n_jobs=-1)
    print(scores)
