import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer

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
            cur = pd.to_datetime(X[i]).dt
            ret_x[str(i) + ' ' + 'Year'] = cur.year
            ret_x[str(i) + ' ' + 'Month'] = cur.month
            ret_x[str(i) + ' ' + 'Day'] = cur.day
            ret_x[str(i) + ' ' + 'Quarter'] = cur.quarter
            ret_x[str(i) + ' ' + 'DaysInMonth'] = cur.daysinmonth
            ret_x[str(i) + ' ' + 'Sec'] = cur.microsecond // 10 ** 9
        return ret_x


pipe = Pipeline([
    ('trans', ColumnTransformer([
        ('num', Pipeline([
            ('empty', FunctionTransformer())
        ]), num_features),
        ('cat', Pipeline([
            ('oh', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), cat_features),
        ('time', Pipeline([
            ('tm', TimeFeatures())
        ]), time_features),
    ])),
    ('reg', LGBMRegressor(silent=True, n_estimators=500)),
])
print(pipe.get_params())
pipe.fit(train_features, train_label)
print([round(i, 2) for i in train_label[:10]])
print([round(i, 2) for i in pipe.predict(train_features[:10])])
print(mean_squared_error(train_label, pipe.predict(train_features), squared=False))

cv_results = cross_validate(pipe, train_features, train_label,
                            scoring='neg_mean_squared_error', cv=5, return_estimator=True)
scores = cv_results["test_score"]
scores = np.sqrt(-scores)
print(np.mean(scores), np.std(scores), list(scores))

best_estimator = cv_results['estimator'][list(scores).index(min(scores))]

print('------------------------------ search parameters ------------------------------')
test_df = start_test_df.copy()
test_id = test_df['Id'].values
test_label = test_df['Sold Price'].values
test_features = pd.DataFrame(test_df.drop(columns=['Id']))
test_result = best_estimator.predict(test_features)

print([round(i, 2) for i in test_label[:10]])
print([round(i, 2) for i in test_result[:10]])
print(mean_squared_error(test_label, test_result, squared=False))

print('------------------------------ submit ------------------------------')
pred_id = pred['Id'].values
pred_features = pd.DataFrame(pred.drop(columns=['Id']))
print(pred_id.shape, pred_features.shape)

submit = []
for (id, y) in zip(pred_id, best_estimator.predict(pred_features)):
    submit.append({'Id': id, 'Sold Price': y})
# noinspection PyTypeChecker
pd.DataFrame(submit).to_csv('submit.csv', index=False)
