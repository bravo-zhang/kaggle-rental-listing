import re
import pandas as pd
from argparse import ArgumentParser
from nltk.stem import PorterStemmer
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer

parser = ArgumentParser(description='Using description text only.')
parser.add_argument('-n', default=2000, type=int, help='Number of boosting rounds.')
parser.add_argument('-c', action='store_true', default=False,
                    help='Set this flag to clean data first.'
                         'Default is cross validation only pre-processed description.')
args = parser.parse_args()

if args.c:
    train = pd.read_json("input/train.json")
    train = train[['listing_id', 'description', 'interest_level']]
    train = train.replace({"interest_level": {"low": 0, "medium": 1, "high": 2}})

    def clean(x):
        stemmer = PorterStemmer()
        regex = re.compile('[^a-zA-Z]')
        i = regex.sub(' ', x).lower().split(" ")
        i = [stemmer.stem(l) for l in i]
        i = " ".join([l.strip() for l in i if len(l) > 2])
        return i

    train['description_new'] = train.description.apply(lambda x: clean(x))
    cvect_desc = CountVectorizer(stop_words='english', max_features=200)
    sparse = cvect_desc.fit_transform(train.description_new)
    # Renaming words to avoid collisions with other feature names in the model
    col_desc = ['desc_' + i for i in cvect_desc.get_feature_names()]
    count_vect_df = pd.DataFrame(sparse.todense(), columns=col_desc)
    train = pd.concat([train.reset_index(), count_vect_df],axis=1)
    train = train.drop(['description', 'index', 'description_new'], axis=1)
    train.to_csv('data/description.csv', index=False)

train = pd.read_csv("data/description.csv")
y_train = train['interest_level']
X_train = train.drop(['interest_level','listing_id'], axis=1)
lgb_train = lgb.Dataset(X_train, y_train)
lgb_param = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_classes': 3,
    'metric': {'multi_logloss'},
    'num_leaves': 55,
    'learning_rate': 0.01,
    'feature_fraction': 0.82,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
cv_results = lgb.cv(lgb_param, lgb_train, num_boost_round=args.n, nfold=5, stratified=True,
                    callbacks=[lgb.callback.print_evaluation(show_stdv=True)])
