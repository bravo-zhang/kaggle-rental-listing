import numpy as np
import pandas as pd
import argparse
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.externals import joblib

parser = argparse.ArgumentParser(description='SVM Model.')
parser.add_argument('-s', action='store_true', default=False,
                    help='Set this flag for submission. Default is cross validation.')
args = parser.parse_args()

X_train = pd.read_csv("data/train_python.csv", encoding='utf-8')
y_train = X_train['interest_level']
X_train = X_train.drop('interest_level', axis=1)
clf = CalibratedClassifierCV(LinearSVC(), method='isotonic')

if args.s:
    clf.fit(X_train, y_train)
    joblib.dump(clf, 'checkpoint/svc.pkl')
    X_test = pd.read_csv("data/test_python.csv", encoding='utf-8')
    pred = clf.predict_proba(X_test)
    np.savetxt('submission/submission.csv', np.c_[X_test['listing_id'], pred[:, [2, 1, 0]]], delimiter=',',
               header='listing_id,high,medium,low', fmt='%d,%.16f,%.16f,%.16f', comments='')
else:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=2018)
    scores = cross_val_score(clf, X_train, y_train, scoring='neg_log_loss', cv=cv, n_jobs=-1)
    print(scores)
