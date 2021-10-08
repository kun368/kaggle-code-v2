from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from utils.common import compress_read, set_pandas_option

print('------------------------------ read inputs ------------------------------')
set_pandas_option()

train = compress_read('data/train.csv')
pred = compress_read('data/test.csv')
sample_submission = compress_read('data/sample_submission.csv')

start_train_df, start_test_df = train_test_split(train, test_size=0.2, random_state=42)
start_train_df, start_test_df = pd.DataFrame(start_train_df), pd.DataFrame(start_test_df)
start_train_df.info()

print('------------------------------ data distribution ------------------------------')
train_df = start_train_df.copy()

for i in train_df.columns:
    counter = Counter(train_df[i].values)
    print(i, len(counter.keys()), counter.most_common(5))

train_df.sample(n=1000).to_excel('sample_data.xlsx', index=False)

train_df.hist(bins=50, figsize=(20, 15))
plt.savefig('plots/hist.png', dpi=600)

corr_mat = train_df.corr()
print(corr_mat['Sold Price'].sort_values(ascending=False))

print('------------------------------ training preparation ------------------------------')
train_df = start_train_df.copy()
train_id = train_df['Id'].values
train_label = train_df['Sold Price'].values
train_features = pd.DataFrame(train_df.drop(columns=['Id', 'Sold Price']))
print(train_id.shape, train_label.shape, train_features.shape)

time_features = ['Year built', 'Listed On', 'Last Sold On']
num_features = []
cat_features = []

for i in train_features.columns:
    if str(i) in time_features:
        continue
    if train_features[i].dtypes == 'float64':
        num_features.append(i)
    else:
        cat_features.append(i)
print('time_features', time_features)
print('num_features', num_features)
print('cat_features', cat_features)


class TimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        ret_x = pd.DataFrame()
        for i in time_features:
            ret_x[str(i) + ' ' + 'Year'] = pd.to_datetime(X[i]).dt.year
        return ret_x


pipe = Pipeline([
    ('imputer_3', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
    ('imputer_1', SimpleImputer(missing_values='None', strategy='constant', fill_value=0)),
    ('imputer_2', SimpleImputer(missing_values='', strategy='constant', fill_value=0)),
    ('trans', ColumnTransformer([
        ('num', Pipeline([
            ('std_scalar', StandardScaler())
        ]), num_features),
        ('cat', Pipeline([
            ('oh', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), cat_features),
        ('time', Pipeline([
            ('tm', TimeFeatures())
        ]), time_features),
    ])),
    ('reg', LGBMRegressor(silent=False, zero_as_missing=True)),
])
print(pipe.get_params())
pipe.fit(train_features, train_label)
print([round(i, 2) for i in train_label[:10]])
print([round(i, 2) for i in pipe.predict(train_features[:10])])
print(mean_squared_error(train_label, pipe.predict(train_features), squared=False))

scores = cross_val_score(pipe, train_features, train_label, scoring='neg_mean_squared_error', cv=5)
scores = np.sqrt(-scores)
print(np.mean(scores), np.std(scores), list(scores))
