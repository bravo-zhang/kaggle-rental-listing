import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.externals import joblib


# X_train = pd.read_csv("train_python.csv", encoding='utf-8')
X_train = pd.read_csv("train_plain_col.csv", encoding='utf-8')
y_train = X_train['interest_level']
X_train = X_train.drop('interest_level', axis=1)

X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, test_size=0.3)

num_rounds = 2000

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

xgtrain = xgb.DMatrix(X_train, label=y_train)
# xgb.cv(xgb_param, xgtrain, num_rounds, nfold=5, stratified=True,
#              callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
clf = xgb.train(xgb_param, xgtrain, num_rounds)
joblib.dump(clf, 'xgb.pkl')

# pred = clf.predict(xgb.DMatrix(X_test))
