from collections import Counter

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer

from utils import common

train = common.compress_read('data/train.csv')
pred = common.compress_read('data/test.csv')
sample_submit = common.compress_read('data/sample_submission.csv')

# basic information
train.info()
pred.info()

print(np.any(pd.isna(train).values))
print(np.any(pd.isna(pred).values))

# data distribution
for i in train.columns:
    cnt = Counter(train[i].values)
    print(i, len(cnt.keys()), cnt.most_common(10))

print(np.std(train['loss']), stats.skew(train['loss']))
print(np.std(np.log(train['loss'])), stats.skew(np.log(train['loss'])))

# training prep

pipe = Pipeline(steps=[
    ('feature_select', FunctionTransformer(lambda x: x.drop(columns=['id', 'loss'], errors='ignore'))),
    ('ord_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
    ('model', LGBMRegressor(silent=False, n_estimators=500, max_depth=10)),
])
pipe = TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)
print(pipe.get_params())

train, test = train_test_split(train, test_size=0.2, random_state=41)
pipe.fit(train, train['loss'])
test_y = test['loss']
test_m = pipe.predict(test)

print([round(i, 2) for i in test_y[:10].values])
print([round(i, 2) for i in test_m[:10]])
print(r2_score(test_y, test_m))
print(mean_absolute_error(test_y, test_m))
print(mean_squared_error(test_y, test_m, squared=True))

pred_m = pipe.predict(pred)
submit = []
for (id, y) in zip(pred['id'], pred_m):
    submit.append({'id': id, 'loss': y})
# noinspection PyTypeChecker
pd.DataFrame(submit).to_csv('submit.csv', index=False)
