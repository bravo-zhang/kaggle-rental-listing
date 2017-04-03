import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import log_loss

X_train = pd.read_csv("train_plain_col.csv", encoding='utf-8')
y_train = X_train['interest_level']
X_train = X_train.drop('interest_level', axis=1)
X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, test_size=0.3)

print len(list(X_train))
print log_loss(y_valid, RandomForestClassifier().fit(X_train_part, y_train_part).predict_proba(X_valid))

param_grid = {'n_estimators': [500, 700, 1000], 'max_features': [16, 17, 18]}
model = GridSearchCV(param_grid=param_grid, n_jobs=2, cv=3, verbose=20, scoring="neg_log_loss",
                     estimator=RandomForestClassifier())
model = model.fit(X_train_part, y_train_part)
print model
print model.cv_results_
print model.best_params_
print log_loss(y_valid, model.predict_proba(X_valid))
