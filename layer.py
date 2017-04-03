import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import log_loss


X_train = pd.read_csv("s_train_part.csv")
y_train = X_train.icol(0)
X_train = X_train.drop(X_train.columns[0], axis=1)
X_valid = pd.read_csv("s_valid.csv")
y_valid = X_valid.icol(0)
X_valid = X_valid.drop(X_valid.columns[0], axis=1)

# param_grid = {'n_estimators': [50, 100], 'max_features': [2, 'auto'], 'learning_rate': [0.05, 0.1, 0.2]}
# model = GridSearchCV(param_grid=param_grid, n_jobs=2, cv=3, verbose=20, scoring="neg_log_loss",
#                      estimator=GradientBoostingClassifier())
# model = model.fit(X_train, y_train)
# print model.best_params_

print log_loss(y_valid, GradientBoostingClassifier(max_features=2, learning_rate=0.05).fit(X_train, y_train).predict_proba(X_valid))
print log_loss(y_valid, GradientBoostingClassifier(max_features=2, learning_rate=0.1).fit(X_train, y_train).predict_proba(X_valid))
print log_loss(y_valid, LogisticRegression().fit(X_train, y_train).predict_proba(X_valid))
# print log_loss(y_valid, XGBClassifier().fit(X_train, y_train).predict_proba(X_valid))
print log_loss(y_valid, GradientBoostingClassifier().fit(X_train, y_train).predict_proba(X_valid))
