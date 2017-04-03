import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.externals import joblib


# X_train = pd.read_csv("train_python.csv", encoding='utf-8')
X_train = pd.read_csv("train_plain_col.csv", encoding='utf-8')
y_train = X_train['interest_level']
X_train = X_train.drop('interest_level', axis=1)

X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, test_size=0.3)

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
    'verbose': 0
}

lgb_train = lgb.Dataset(X_train_part, y_train_part)
lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
gbm = lgb.train(lgb_param, lgb_train, num_boost_round=num_rounds, valid_sets=lgb_valid, early_stopping_rounds=10)
y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
print log_loss(y_valid, y_pred)

# lgb_train = lgb.Dataset(X_train, y_train)
# gbm = lgb.train(lgb_param, lgb_train, num_boost_round=num_rounds)
# joblib.dump(gbm, 'lgb.pkl')
