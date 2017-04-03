import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.externals import joblib
from ensemble import Ensemble


# X_train = pd.read_csv("train_python.csv", encoding='utf-8')
X_train = pd.read_csv("train_plain_col.csv", encoding='utf-8')
y_train = X_train['interest_level']
X_train = X_train.drop('interest_level', axis=1)

rf_model = RandomForestClassifier(n_estimators=500, max_features=18, max_depth=22, random_state=2017)
xgb_model = XGBClassifier()
lgb_model = LGBMClassifier()
gbt_model = GradientBoostingClassifier(random_state=2017)
ensemble = Ensemble(5, gbt_model, [lgb_model, xgb_model, rf_model])

# Self cross validation
X_train_part, X_valid, y_train_part, y_valid = \
    train_test_split(X_train, y_train, test_size=0.3, random_state=2017)
y_pred = ensemble.fit_predict_proba(X_train_part, y_train_part, X_valid, y_valid)
print log_loss(y_valid, y_pred)

# Generate submission csv
# X_test = pd.read_csv("test_python.csv")
# y_pred = ensemble.fit_predict_proba(X_train, y_train, X_test)
# np.savetxt('submission.csv', np.c_[X_test['listing_id'], y_pred[:, [2, 1, 0]]], delimiter=',',
#            header='listing_id,high,medium,low', fmt='%d,%.16f,%.16f,%.16f', comments='')
